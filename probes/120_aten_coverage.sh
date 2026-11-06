set -u
# 探针 120：aten 分发层第二批的「覆盖 + 与 CPU 对齐」总表（沙箱真机）。
#
# 为什么要有这一条：下一批算子里有一大半**上游已经用 composite 实现好了**（dispatch 表里
# `Undefined: registered ... [default backend kernel]` 会被复制到 PrivateUse1 键上），
# 自己再 impl 一遍会盖住上游语义、升级后行为反而变差。所以判据不能是「我能不能写」，
# 只能是「实测它到底缺不缺、缺了以后算得对不对」。本探针同时给出：
#   算子 / 是否可用 / 走的是 composite 还是我们的实现 / 形状·stride·dtype 是否与 CPU 一致 /
#   max_abs_diff
#
# 两种用法：
#   基线（_aten_extra 还没写）：只看哪些 FAIL；
#   验收（_aten_extra 就位）：看 FAIL 有没有消失、数值有没有对齐。
#
# 前置：src/ 已推到 $W/src（含 _corex_shim.so），triton 3.4 在 venv 里。
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/120_aten_coverage.sh --name 120_aten_coverage --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/aten_coverage.py" <<'PYEOF'
"""逐个算子在 corex 与 CPU 上跑同一件事，比对形状 / stride / dtype / 数值。"""
import itertools
import sys
import traceback

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

DEV = "corex"

import corex_torch
from corex_torch import _runtime as rt
from corex_torch._device import _scalar_from_stage

corex_torch.register()

try:
    from corex_torch import _aten_extra
except ImportError:
    _aten_extra = None
OURS = set(getattr(_aten_extra, "REGISTERED", ())) if _aten_extra else set()
print(f"torch {torch.__version__} | backend name = {torch._C._get_privateuse1_backend_name()}")
print(f"_aten_extra = {'已加载，注册 ' + str(len(OURS)) + ' 个算子' if _aten_extra else '未提供（基线）'}")

# 部署的 _device.py 可能还是「没有 _aten_extra 桥接」的那一版（它在被持续编辑）。
# 为了不把验收绑在别人文件的版本上，这里自己开一个 Library 并显式 register——
# 走的是同一套 `lib.impl(..., "PrivateUse1")`，与桥接等价。
if _aten_extra is not None and not OURS:
    import inspect

    import corex_torch._device as _dev

    wired = "_aten_extra" in inspect.getsource(_dev._register_ops)
    print(f"  _device._register_ops 里有桥接: {wired}；本轮改用探针自己注册")
    _lib = torch.library.Library("aten", "IMPL")
    _aten_extra.register(_lib)
    OURS = set(getattr(_aten_extra, "REGISTERED", ()))
    print(f"  探针显式注册后 OURS = {len(OURS)} 个")

# fp32 的判据：单次逐元素运算应当逐元素相等；规约类因求和顺序不同留 1e-6 余量。
# fp32 的 eps = 2^-23 ≈ 1.19e-7，量级 10 以内的 6~36 个元素重排误差远小于 1e-6。
TOL = {torch.float32: 1e-6, torch.float64: 1e-12, torch.float16: 1e-3,
       torch.bfloat16: 1e-2}


def base(dev, shape=(2, 3), dtype=torch.float32, offset=0.5):
    """造一个同值的参考张量：先 CPU 造，再搬到目标设备（只依赖 empty + _copy_from）。"""
    n = 1
    for s in shape:
        n *= int(s)
    cpu = torch.arange(n, dtype=dtype).reshape(tuple(shape)) + offset
    return cpu if dev == "cpu" else cpu.to(DEV)


def mask(dev, shape=(2, 3)):
    """布尔掩码，CPU 造完搬过去。"""
    m = torch.zeros(shape, dtype=torch.bool)
    m[0, 0] = True
    m[1, 2] = True
    return m if dev == "cpu" else m.to(DEV)


def to_cpu(t):
    """把 corex 张量读回 CPU。

    连续张量直接 `.cpu()`；非连续张量**按 stride 逐元素读**——因为 `_copy_from` 目前是
    线性 memcpy（只校验 shape 不校验 stride），拿它读非连续张量会把别的元素读回来。
    """
    if t.device.type != DEV:
        return t
    if t.numel() == 0:
        return torch.empty(tuple(t.shape), dtype=t.dtype)
    if t.is_contiguous():
        return t.cpu()
    rt.set_device(int(t.device.index or 0))
    item = int(t.element_size())
    stage = rt.staging(item)
    vals = []
    for idx in itertools.product(*[range(int(s)) for s in t.shape]):
        off = sum(int(i) * int(st) for i, st in zip(idx, t.stride())) * item
        rt.memcpy(stage, t.data_ptr() + off, item, rt.MEMCPY_DEVICE_TO_HOST)
        vals.append(_scalar_from_stage(stage, t.dtype))
    return torch.tensor(vals, dtype=t.dtype).reshape(tuple(t.shape))


def cmp_tensor(ref, got):
    """比一个张量：形状 / stride / dtype / 数值。

    **stride 必须拿原始 corex 张量比**：读回 CPU 的张量一定是连续的（`_copy_from` 写进
    `empty_like` 的结果），拿它当 stride 基准会把「视图正确但读回物化」误判成不符。

    Returns:
        (ok, 说明串, max_abs_diff 或 None)
    """
    if not isinstance(got, torch.Tensor):
        return False, f"类型不同 {type(got).__name__}", None
    got_cpu = to_cpu(got)
    if tuple(ref.shape) != tuple(got_cpu.shape):
        return False, f"shape {tuple(ref.shape)} vs {tuple(got_cpu.shape)}", None
    if tuple(ref.stride()) != tuple(got.stride()):
        return False, f"stride {tuple(ref.stride())} vs {tuple(got.stride())}", None
    if ref.dtype != got_cpu.dtype:
        return False, f"dtype {ref.dtype} vs {got_cpu.dtype}", None
    if ref.numel() == 0:
        return True, "empty", 0.0
    if ref.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        diff = (ref.double() - got_cpu.double()).abs().max().item()
        tol = TOL.get(ref.dtype, 1e-6)
        return diff <= tol, f"max_abs_diff={diff:.3e} (tol={tol:g})", diff
    same = bool(torch.equal(ref, got_cpu))
    return same, "exact" if same else "整数/布尔不等", 0.0 if same else None


def cmp(ref, got):
    """递归比较（支持 tuple / list / 标量 / 张量）。"""
    if isinstance(ref, (tuple, list)):
        if not isinstance(got, (tuple, list)) or len(ref) != len(got):
            return False, f"结构不同 {type(ref).__name__} vs {type(got).__name__}", None
        details, worst = [], None
        for a, b in zip(ref, got):
            ok, msg, diff = cmp(a, b)
            details.append(msg)
            if diff is not None:
                worst = diff if worst is None else max(worst, diff)
            if not ok:
                return False, "; ".join(details), worst
        return True, "; ".join(details), worst
    if isinstance(ref, torch.Tensor):
        return cmp_tensor(ref, got)
    same = ref == got
    return bool(same), f"{ref!r} vs {got!r}", 0.0 if same else None


# --- 每个 case：(名字, 在某个设备上跑出结果的可调用对象) ---
def cases():
    """返回 [(算子名, fn(dev) -> 结果)]。每个 fn 内部自造输入，避免互相污染。"""
    t = torch
    return [
        # 1) 元数据 / 形状 / 视图
        ("view", lambda d: base(d, (2, 3)).view(6)),
        ("_unsafe_view", lambda d: t.ops.aten._unsafe_view(base(d, (2, 3)), [6])),
        ("view_as", lambda d: base(d, (2, 3)).view_as(base(d, (3, 2)))),
        ("t", lambda d: base(d, (2, 3)).t()),
        ("transpose.int", lambda d: base(d, (2, 3)).transpose(0, 1)),
        ("permute", lambda d: base(d, (2, 3)).permute(1, 0)),
        ("squeeze", lambda d: base(d, (1, 2, 3)).squeeze()),
        ("squeeze.dim", lambda d: base(d, (1, 2, 3)).squeeze(0)),
        ("unsqueeze", lambda d: base(d, (2, 3)).unsqueeze(0)),
        ("expand", lambda d: base(d, (2, 1)).expand(2, 3)),
        ("expand_as", lambda d: base(d, (2, 1)).expand_as(base(d, (2, 3)))),
        ("detach", lambda d: base(d, (2, 3)).detach()),
        ("alias", lambda d: t.ops.aten.alias(base(d, (2, 3)))),
        ("contiguous", lambda d: base(d, (2, 3)).t().contiguous()),
        ("clone", lambda d: base(d, (2, 3)).clone()),
        ("clone_of_view", lambda d: base(d, (2, 3)).t().clone()),
        ("select.int", lambda d: base(d, (2, 3)).select(0, 1)),
        ("slice.Tensor", lambda d: base(d, (2, 3))[0:2]),
        ("narrow", lambda d: base(d, (2, 3)).narrow(1, 0, 2)),
        ("as_strided", lambda d: t.as_strided(base(d, (2, 3)), (2, 3), (1, 2))),
        # 2) 规约 / 激活 / in-place
        ("sum", lambda d: base(d, (2, 3)).sum()),
        ("sum.dim_IntList", lambda d: base(d, (2, 3)).sum(0)),
        ("sum.IntList_out", lambda d: t.sum(base(d, (2, 3)), [0],
                                            out=t.zeros(3, device=d))),
        ("mean.dim", lambda d: base(d, (2, 3)).mean(0)),
        ("max.dim", lambda d: base(d, (2, 3)).max(0)),
        ("argmax", lambda d: base(d, (2, 3)).argmax(0)),
        ("softmax.int", lambda d: t.softmax(base(d, (2, 3)), 1)),
        ("_softmax", lambda d: t.ops.aten._softmax(base(d, (2, 3)), 1, False)),
        ("log_softmax.int", lambda d: t.log_softmax(base(d, (2, 3)), 1)),
        ("var.correction", lambda d: base(d, (2, 3)).var(0)),
        ("rsqrt", lambda d: base(d, (2, 3)).rsqrt()),
        ("relu", lambda d: t.relu(base(d, (2, 3), offset=-2.0))),
        ("relu_", lambda d: base(d, (2, 3), offset=-2.0).relu_()),
        ("silu", lambda d: t.nn.functional.silu(base(d, (2, 3)))),
        ("gelu", lambda d: t.nn.functional.gelu(base(d, (2, 3)))),
        ("add_.Tensor", lambda d: base(d, (2, 3)).add_(base(d, (2, 3)))),
        ("mul_.Tensor", lambda d: base(d, (2, 3)).mul_(base(d, (2, 3)))),
        ("add_.Scalar", lambda d: base(d, (2, 3)).add_(2.0)),
        # 3) 拼装 / 索引
        ("cat", lambda d: t.cat([base(d, (2, 3)), base(d, (2, 3))], 0)),
        ("stack", lambda d: t.stack([base(d, (2, 3)), base(d, (2, 3))], 0)),
        ("index_select", lambda d: t.index_select(
            base(d, (2, 3)), 0, t.tensor([1, 0], dtype=torch.int64,
                                         device="cpu" if d == "cpu" else DEV))),
        ("embedding", lambda d: t.nn.functional.embedding(
            t.tensor([[0, 2], [1, 3]], dtype=torch.int64,
                     device="cpu" if d == "cpu" else DEV), base(d, (4, 3)))),
        ("masked_fill_.Scalar", lambda d: base(d, (2, 3)).masked_fill_(mask(d), -1.0)),
    ]


print()
print(f"{'算子':22s} {'结果':6s} {'路由':10s} 说明")
print("-" * 118)
rows = []
for name, fn in cases():
    route = "ours" if name in OURS else "composite/上游"
    try:
        ref = fn("cpu")
    except Exception as exc:                            # noqa: BLE001 - CPU 侧不该失败
        print(f"{name:22s} {'CPU失败':6s} {'-':10s} {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        rows.append((name, "CPU_FAIL", "-", ""))
        continue
    try:
        got = fn(DEV)
    except Exception as exc:                            # noqa: BLE001 - 这里就是要抓缺口
        first = str(exc).splitlines()[0][:110]
        print(f"{name:22s} {'FAIL':6s} {route:10s} {type(exc).__name__}: {first}")
        rows.append((name, "FAIL", route, f"{type(exc).__name__}: {first}"))
        continue
    try:
        ok, msg, diff = cmp(ref, got)
    except Exception as exc:                            # noqa: BLE001 - 比较本身也可能踩到缺口
        print(f"{name:22s} {'CMP?':6s} {route:10s} 比较失败 {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        rows.append((name, "CMP_ERR", route, str(exc).splitlines()[0][:80]))
        continue
    print(f"{name:22s} {'OK' if ok else 'MISMATCH':6s} {route:10s} {msg}")
    rows.append((name, "OK" if ok else "MISMATCH", route, msg))

ok_n = sum(1 for r in rows if r[1] == "OK")
fail = [r[0] for r in rows if r[1] == "FAIL"]
bad = [r[0] for r in rows if r[1] in ("MISMATCH", "CMP_ERR")]
print()
print(f"=== 汇总：OK {ok_n}/{len(rows)}；FAIL {len(fail)}；数值不符 {len(bad)} ===")
print("FAIL     :", fail)
print("MISMATCH :", bad)

# 给下一批留的清单：这些算子此刻真的缺（不是 composite），要显式实现或转给内核组。
print()
print("MISSING_OPS =", fail)
print("ATEN_COVERAGE_DONE")
raise SystemExit(0 if not bad else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/aten_coverage.py" "$W/src"
RC=$?
echo "ATEN_COVERAGE_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE120_DONE $(date -Is)"
exit $RC
