set -u
# 目的：定位「torch 端到端 341 µs vs 内核本身 178 µs」那 163 µs 差额花在哪。
# 为什么值得单独一个探针：coffin 的 decode 每 token 有六百多次发射（GRAPH-REPLAY.md §17.1），
# 若每次多 163 µs，一步就是 100 ms 量级——比 kernel 本身还贵。所以差额必须在接引擎前算清。
#
# 拆成四段分别计时：
#   A. 只有内核 + 启动（直接调 _ops_matmul.mm，指针都是现成的，不经 aten）
#   B. 小 kernel（16³）——把内核时间榨到接近 0，剩下的就是启动地板
#   C. 分配（shim.empty_strided）——aten 的每个算子都要先拿输出张量
#   D. 完整 aten 路径（a @ b）
# 每段都在循环外同步一次，避免把等待算进单次耗时。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/overhead.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch
from corex_torch import _aten_kernels, _device, _ops_matmul, _ops_reduce, _triton

corex_torch.register()
shim = _device.shim()


def bench(label, fn, reps, unit="us"):
    """跑 reps 次，循环外同步一次；返回单次耗时。第一次单独报（含冷启动）。"""
    fn()
    torch.corex.synchronize()
    t0 = time.perf_counter()
    first = 0.0
    for i in range(reps):
        t = time.perf_counter()
        fn()
        if i == 0:
            first = time.perf_counter() - t
    torch.corex.synchronize()
    total = time.perf_counter() - t0
    per = total / reps
    scale = 1e6 if unit == "us" else 1e3
    print(f"  {label:46s} {per * scale:9.2f} {unit}/次   首次 {first * scale:9.2f} {unit}")
    return per


def main():
    M = N = K = 256
    a = torch.randn(M, K).to("corex")
    b = torch.randn(K, N).to("corex")
    c = torch.empty(M, N, device="corex")
    pa, pb, pc = a.data_ptr(), b.data_ptr(), c.data_ptr()

    print("=== 预热（编译内核，避免把编译算进来）===")
    _ops_matmul.mm(pa, pb, pc, M, N, K, "fp32")
    torch.corex.synchronize()
    print("  内核已编译，缓存：", _ops_matmul.cache_info()["kernels"], "个")

    print("=== A. 只有内核 + 启动（不经 aten，指针现成）===")
    per_a = bench("mm 256 直接调 _ops_matmul.mm", lambda: _ops_matmul.mm(pa, pb, pc, M, N, K, "fp32"), 20)
    gflops = 2 * M * N * K / per_a / 1e9
    print(f"       -> {gflops:.1f} GFLOP/s（内核 178 µs / 188.8 GFLOP/s 那个口径含 L2 刷）")

    print("=== B. 小内核（16³，把内核时间榨到接近 0）===")
    m2 = 16
    a2 = torch.randn(m2, m2).to("corex")
    b2 = torch.randn(m2, m2).to("corex")
    c2 = torch.empty(m2, m2, device="corex")
    _ops_matmul.mm(a2.data_ptr(), b2.data_ptr(), c2.data_ptr(), m2, m2, m2, "fp32")
    torch.corex.synchronize()
    per_b = bench("mm 16 直接调（≈ 纯启动地板）", lambda: _ops_matmul.mm(
        a2.data_ptr(), b2.data_ptr(), c2.data_ptr(), m2, m2, m2, "fp32"), 200)
    print(f"       -> 启动地板约 {per_b:.1f} µs/次")

    print("=== C. 分配（aten 每个算子都要先拿输出张量）===")
    code = _device._scalar_type(torch.float32)
    bench("shim.empty_strided([256,256],[256,1])", lambda: shim.empty_strided([M, N], [N, 1], code, 0), 200)

    print("=== D. 完整 aten 路径 ===")
    per_d = bench("a @ b（aten 分发 + 分配 + 内核）", lambda: a @ b, 20)
    print(f"       -> {2 * M * N * K / per_d / 1e9:.1f} GFLOP/s")

    print("=== E. 差额归因 ===")
    print(f"  aten 端到端 {per_d * 1e6:.1f} us - 直接调内核 {per_a * 1e6:.1f} us = "
          f"{(per_d - per_a) * 1e6:.1f} us（分配 + 分发 + Python 层）")
    print(f"  纯内核时间 ≈ {max(0.0, per_a - per_b) * 1e6:.1f} us（256³）")

    print("=== F. 归约类看一眼（softmax 32x512）===")
    x = torch.randn(32, 512).to("corex")
    o = torch.empty(32, 512, device="corex")
    _ops_reduce.softmax_last(x.data_ptr(), o.data_ptr(), 32, 512, "fp32")
    torch.corex.synchronize()
    bench("softmax 直接调", lambda: _ops_reduce.softmax_last(
        x.data_ptr(), o.data_ptr(), 32, 512, "fp32"), 200)

    print("OVERHEAD_DONE")


main()
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/overhead.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
