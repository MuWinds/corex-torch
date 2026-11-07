set -u
# 探针 121：视图语义（共享存储 / 写穿 / storage_offset）与我们注册的每条算子的直调验证。
#
# 为什么 120 不够：120 是「算子对外表现」的黑盒对比，能告出「缺」但分不清「缺」还是
# 「我的实现没被调到」。本探针改从两个角度看：
#   1. **视图语义的硬要求**——`as_strided` 必须共享同一块存储：透过视图写，基张量要能看见；
#      `storage_offset` 必须真的生效；视图存活期间底层存储不能被释放。
#      这三条只有真机验得了（黑盒数值对比看不出来）。
#   2. **逐条直调我们注册的算子**：绕开 composite，`torch.ops.aten.xxx(...)` 直接打进去，
#      确认走的是我们的实现，并单独量数值。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/121_view_alias.sh --name 121_view_alias --tail 0

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

cat > "$W/view_alias.py" <<'PYEOF'
"""视图别名 + 我们那条注册的直接验证。"""
import ctypes
import sys

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

DEV = "corex"
import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _aten_extra

corex_torch.register()
# 部署的 _device.py 可能还没带桥接，这里显式注册一次（与桥接等价）。
LIB = torch.library.Library("aten", "IMPL")
_aten_extra.register(LIB)

print("REGISTERED =", _aten_extra.REGISTERED)
print("warnings   =", _aten_extra.warnings() or "（无）")

ok = True


def check(label, cond, detail=""):
    """打印一条断言。"""
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {label}{('  ' + detail) if detail else ''}")
    ok = ok and bool(cond)


print("\n=== 1. shim.as_strided：共享存储 / stride / storage_offset ===")
shim = corex_torch._device.shim()
base = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(DEV)   # 3x4 连续
print("  base:", tuple(base.shape), tuple(base.stride()), "ptr=0x%x" % base.data_ptr())
view = shim.as_strided(base, [4, 3], [1, 4], 0)                     # 转置视图
check("视图形状", tuple(view.shape) == (4, 3), str(tuple(view.shape)))
check("视图 stride", tuple(view.stride()) == (1, 4), str(tuple(view.stride())))
check("data_ptr 相同（同一块存储）", view.data_ptr() == base.data_ptr())
check("storage 是同一个", view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr())

# storage_offset：跳过前 2 个元素再看
off = shim.as_strided(base, [2, 4], [4, 1], 2)
check("storage_offset 生效", int(off.storage_offset()) == 2, str(int(off.storage_offset())))

print("\n=== 2. 写穿：透过视图改，基张量必须看得见 ===")
# 把视图 [1]（= base.flatten()[1]）写成 0，再读 base 看是否被改到
one = shim.as_strided(base, [1], [1], 1)
torch.ops.aten.zero_(one)
zeros = torch.tensor([0.0], dtype=torch.float32).to(DEV)
from corex_torch._device import _copy_from_impl
_copy_from_impl(zeros, one)                                        # 往视图里写 0
rt.synchronize()
flat = torch.empty(12, dtype=torch.float32)
_copy_from_impl(base.reshape(12), flat)                            # 读回整块
values = [round(float(v), 3) for v in flat]
print("  基张量读回 =", values)
check("写穿到基张量", values[1] == 0.0, f"期望 [0]=0，实际 {values[1]}")
check("其它元素没被碰", values[0] == 0.0 and values[2] == 2.0)

print("\n=== 3. 视图 alive 时底层存储不能被释放（生命周期）===")
before = shim.allocated_bytes()
tmp = torch.arange(8, dtype=torch.float32).to(DEV)
tmp_view = shim.as_strided(tmp, [8], [1], 0)
del tmp                                     # 只删基张量，视图还活着
import gc

gc.collect()
alive = shim.allocated_bytes()
check("视图持有存储所有权", alive >= before, f"allocated {before} -> {alive}")
vals = torch.empty(8, dtype=torch.float32)
_copy_from_impl(tmp_view, vals)
check("删除基张量后视图仍可读", [float(v) for v in vals] == [float(i) for i in range(8)])
del tmp_view
gc.collect()
check("视图释放后账本回落", shim.allocated_bytes() <= alive)

def base_div(d):
    """除法的除数：在 CPU 上造好再搬过去（corex 上 `+ 1.0` 目前会撞 add.out 的标量路径）。"""
    return (torch.arange(6, dtype=torch.float32).reshape(2, 3) + 2.0).to(d)


print("\n=== 3.5 contiguous / clone 对非连续源的处理（`_copy_from` 是线性 memcpy，最容易错在这）===")
src = base.clone().t()                       # (4,3) 的转置视图，stride=(1,4)
print("  src stride =", tuple(src.stride()), "| contiguous =", src.is_contiguous())
for label, fn in (("contiguous()", lambda t: t.contiguous()),
                  ("clone()", lambda t: t.clone())):
    try:
        got = fn(src)
    except Exception as exc:                 # noqa: BLE001
        print(f"  FAIL {label}: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        ok = False
        continue
    back = torch.empty(tuple(got.shape), dtype=torch.float32)
    _copy_from_impl(got, back)
    want = fn(base.cpu().t())
    good = torch.equal(back, want)
    print(f"  {'OK  ' if good else 'FAIL'} {label}: stride={tuple(got.stride())} "
          f"读回[0]={back.flatten()[:4].tolist()} 期望[0]={want.flatten()[:4].tolist()}")
    ok = ok and good

print("\n=== 3.6 copy_ 到底有没有被调到（dispatch 表 + 直接调用）===")
dump = torch._C._dispatch_dump_table("aten::copy_.Tensor")
p1_lines = [line for line in dump.splitlines() if line.startswith("PrivateUse1")]
print("  dispatch 表里的 PrivateUse1 行:", p1_lines or "（没有）")
dst = torch.empty(4, 3, dtype=torch.float32, device=DEV)
torch.ops.aten.copy_.Tensor(dst, src)            # 直接点名，绕开 composite
back = torch.empty(4, 3, dtype=torch.float32)
_copy_from_impl(dst, back)
want = base.cpu().t().contiguous()
good = torch.equal(back, want)
tag = "OK  " if good else "FAIL"
print(f"  {tag} 直接调 copy_.Tensor: 读回[0]={back.flatten()[:4].tolist()} "
      f"期望[0]={want.flatten()[:4].tolist()}")
ok = ok and good

print("\n=== 4. aten 层：直接点名调我们的实现 ===")
cases = [
    ("aten::view", lambda x: torch.ops.aten.view(x, [6])),
    ("aten::_reshape_alias", lambda x: torch.ops.aten._reshape_alias(x, [6], [1])),
    ("aten::mul_.Tensor", lambda x: torch.ops.aten.mul_.Tensor(x, x.clone())),
    ("aten::div_.Tensor", lambda x: torch.ops.aten.div_.Tensor(x, base_div(DEV))),
    ("aten::add_.Scalar", lambda x: torch.ops.aten.add_.Scalar(x, 2.0)),
    ("aten::sub_.Scalar", lambda x: torch.ops.aten.sub_.Scalar(x, 1.0)),
    ("aten::mul_.Scalar", lambda x: torch.ops.aten.mul_.Scalar(x, 3.0)),
    ("aten::div_.Scalar", lambda x: torch.ops.aten.div_.Scalar(x, 2.0)),
]
for name, fn in cases:
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(DEV)
    ref = src.cpu()
    try:
        got = fn(src)
    except Exception as exc:                        # noqa: BLE001
        print(f"  FAIL {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        ok = False
        continue
    back = torch.empty(tuple(got.shape), dtype=torch.float32)
    _copy_from_impl(got, back.reshape(tuple(got.shape)))
    if name.endswith((".Tensor", ".Scalar")):
        # 就地算子：CPU 侧用同一表达式算出参考值
        expr = {"aten::mul_.Tensor": lambda t: t * t.clone(),
                "aten::div_.Tensor": lambda t: t / (torch.arange(6, dtype=torch.float32).reshape(2, 3) + 2.0),
                "aten::add_.Scalar": lambda t: t + 2.0,
                "aten::sub_.Scalar": lambda t: t - 1.0,
                "aten::mul_.Scalar": lambda t: t * 3.0,
                "aten::div_.Scalar": lambda t: t / 2.0}[name]
        want = expr(ref)
        diff = (want - back).abs().max().item()
        good = diff <= 1e-6 and tuple(got.shape) == tuple(want.shape)
        print(f"  {'OK  ' if good else 'FAIL'} {name}: max_abs_diff={diff:.3e} "
              f"shape={tuple(got.shape)}")
        ok = ok and good
    else:
        good = tuple(back.shape) == (6,) and bool(torch.equal(back.reshape(-1), ref.reshape(-1)))
        print(f"  {'OK  ' if good else 'FAIL'} {name}: shape={tuple(got.shape)} "
              f"stride={tuple(got.stride())}")
        ok = ok and good

print("\n=== 5. silu.out 与 cat.out（含 dim=1）===")
x = (torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.0).to(DEV)
out = torch.empty(2, 3, dtype=torch.float32, device=DEV)
torch.ops.aten.silu.out(x, out=out)
back = torch.empty(2, 3, dtype=torch.float32)
_copy_from_impl(out, back)
want = torch.nn.functional.silu(torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.0)
diff = (want - back).abs().max().item()
check("silu.out 数值", diff <= 1e-6, f"max_abs_diff={diff:.3e}")

a = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(DEV)
b = (torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100).to(DEV)
for dim, shape in ((0, (4, 3)), (1, (2, 6))):
    o = torch.empty(shape, dtype=torch.float32, device=DEV)
    torch.ops.aten.cat.out([a, b], dim, out=o)
    got = torch.empty(shape, dtype=torch.float32)
    _copy_from_impl(o, got)
    want = torch.cat([a.cpu(), b.cpu()], dim)
    good = torch.equal(got, want)
    check(f"cat.out dim={dim}", good, "逐元素相等" if good else f"不等 {got.flatten()[:4]} vs {want.flatten()[:4]}")

print("\n=== 6. 结论 ===")
print("VIEW_ALIAS_PASS" if ok else "VIEW_ALIAS_FAIL")
raise SystemExit(0 if ok else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/view_alias.py" "$W/src"
RC=$?
echo "VIEW_ALIAS_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE121_DONE $(date -Is)"
exit $RC
