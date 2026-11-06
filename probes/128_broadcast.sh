set -u
# 探针 128：逐元素快路径的**广义广播**（物化）——正确性（逐字节）+ p50 成本。
#
# 背景（Lead 的口径）：`.out` 的实参可能是广播视图（`x1 * cos[:, None, :]`）。内核按线性
# 下标写、不认广播，所以要么物化、要么报错。**报错会把「能算但慢」变成「不能算」**，那是
# 拿正确性换速度，所以选物化：
#   1. 分配一块与输出同形的连续临时张量（走池分配器，不每次 cudaMalloc）；
#   2. 按广播规则把源填进去（单广播维：每个广播下标一条 memcpy2D）；
#   3. 再走「同形」快路径。
#
# 注意：`.out` 目前仍由 `_device` 的 Python 实现接管（Lead 的探测要等本探针通过才让位），
# 所以这里**用 inplace 形态验证 C++ 的广播物化**（`*_.Tensor` 已由 C++ 接管）。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/128_broadcast.sh --name 128_broadcast --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
export COREX_FAST_OPS=1
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/broadcast.py" <<'PYEOF'
"""广播物化：正确性（逐字节）+ p50 成本，走 C++ 已接管的 inplace 路径。"""
import sys
import time

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _ops_elementwise as elt
from corex_torch import _runtime as rt

corex_torch.register()
shim = corex_torch._device.shim()
copy_from = corex_torch._device._copy_from_impl

supports = (shim.elementwise_supports_broadcast()
            if hasattr(shim, "elementwise_supports_broadcast") else "<旧 .so，没有这个入口>")
print("能力声明 elementwise_supports_broadcast():", supports)

# 手动登记（Lead 的 _register_fast_ops 全量生效后这步由他做；这里保证能验 C++）
if hasattr(elt, "fast_variants") and hasattr(elt, "get_launch_meta"):
    for op, variant in elt.fast_variants():
        if op not in getattr(elt, "FASTPATH_OPS", ()):
            continue
        for dtype in ("fp32", "fp16", "bf16"):
            try:
                meta = elt.get_launch_meta(op, dtype, variant)
                shim.load_fast_op(op, variant, dtype, meta["cubin"], int(meta["num_warps"]),
                                  int(meta["num_stages"]), int(meta["n_ptrs"]),
                                  int(meta["n_i32"]), int(meta["n_f32"]),
                                  bool(meta["alpha_effective"]), int(meta["block"]),
                                  meta["name"], int(meta.get("shared", 0)))
            except Exception:                       # noqa: BLE001 - 编不出来就跳过该组合
                pass
print("已登记逐元素内核条数:", len(shim.elementwise_kernel_info()))

ITERS = 500
failures = []


def bench_p50(fn, iters=ITERS):
    fn()
    rt.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    rt.synchronize()
    samples.sort()
    return samples[len(samples) // 2]


def readback(tensor):
    out = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
    copy_from(tensor, out)
    return out


def case(label, base_cpu, other, op, expect_cpu, limit_us=None):
    """正确性用「重置后一次调用」，计时用「就地累加」（把重置排除在外）。"""
    src = base_cpu.to("corex")
    buf = torch.empty(tuple(base_cpu.shape), dtype=base_cpu.dtype, device="corex")

    def reset():
        copy_from(src, buf)

    try:
        reset()
        op(buf, other)
        got = readback(buf)
        ok = (got.view(torch.uint8).numpy().tobytes()
              == expect_cpu.contiguous().view(torch.uint8).numpy().tobytes())
        p50 = bench_p50(lambda: op(buf, other))
    except Exception as exc:                        # noqa: BLE001
        print(f"  FAIL {label:42s} {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        failures.append(label)
        return
    over = limit_us is not None and p50 > limit_us
    verdict = "OK  " if ok and not over else "FAIL"
    note = "" if ok else "  ← 逐字节不符"
    if over:
        note += f"  ← 超过 {limit_us} us"
    limit = "" if limit_us is None else f" 门槛={limit_us}us"
    print(f"  {verdict} {label:42s} p50={p50:7.2f} us{limit}{note}")
    if not ok or over:
        failures.append(label)


mul_ = torch.Tensor.mul_


def main():
    B, T, H = 4, 16, 64
    torch.manual_seed(7)
    x_cpu = torch.randn(B, T, H) * 0.3
    cos_cpu = torch.randn(B, H) * 0.5
    bias_cpu = torch.randn(H) * 0.2

    print("\n=== 1. 单广播维 [4,16,64] *= [4,1,64]（decode 的 cos[:, None, :] 形态）===")
    cos = cos_cpu.to("corex")[:, None, :]
    case("[4,16,64] *= [4,1,64]（fp32）", x_cpu, cos, mul_, x_cpu * cos_cpu[:, None, :], 20)
    # fp16 的参照必须用「先舍入到 fp16 再相乘」——设备上是 fp16 输入、fp32 中间结果、
    # 存回 fp16；直接拿 fp32 乘积再转 fp16 会有 1 ulp 差异（这是参照写错，不是内核错）。
    x_h = x_cpu.to(torch.float16)
    cos_h = cos_cpu.to(torch.float16)
    case("[4,16,64] *= [4,1,64]（fp16）", x_h, cos_h.to("corex")[:, None, :], mul_,
         x_h * cos_h[:, None, :], 20)

    print("\n=== 2. 末维广播 [4,16,64] *= [64] ===")
    case("[4,16,64] *= [64]（fp32）", x_cpu, bias_cpu.to("corex"), mul_, x_cpu * bias_cpu, 20)

    print("\n=== 3. 标量形态（对照，应该同样快）===")
    case("[4,16,64].mul_(2.0)（0 维 CPU 标量）", x_cpu, 2.0, mul_, x_cpu * 2.0, 20)

    print("\n=== 4. 结论 ===")
    print(f"  用例数 4，失败 {len(failures)}" + (f"：{failures}" if failures else ""))
    print(f"  能力声明 = {supports}")
    ok_all = not failures and supports is True
    print("BROADCAST_PASS" if ok_all else "BROADCAST_FAIL")
    print("BROADCAST_DONE")
    raise SystemExit(0 if ok_all else 1)


main()
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/broadcast.py" "$W/src"
RC=$?
echo "BROADCAST_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE128_DONE $(date -Is)"
exit $RC
