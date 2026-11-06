set -u
# 目的：`_register_fast_ops()` 打开后会不会拖慢初始化？它要在注册期串行编译
# 4 op × 2 variant × 3 dtype = 24 个内核。这份探针量三件事：
#   1. `corex_torch.register()` 本身多久（含 mm 的三个内核）；
#   2. 24 组 `get_launch_meta()` 的**首轮**耗时（真编译）与**第二轮**耗时（磁盘缓存命中）；
#   3. 顺带把 C++ 侧要的字段打印出来，确认表是「数出来的」而不是猜的。
# 结论会决定 `ENABLE_FAST_OPS` 的默认值要不要改成按需。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/fastop_cost.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch
from corex_torch import _device
from corex_torch import _ops_elementwise as elt

t0 = time.perf_counter()
corex_torch.register()
reg_s = time.perf_counter() - t0
print(f"register() = {reg_s * 1000:.0f} ms（含 mm 内核与 fast-op 能力探测）")

shim = _device.shim()
print("shim 导出数 =", len([n for n in dir(shim) if not n.startswith("_")]),
      " hasattr load_fast_op =", hasattr(shim, "load_fast_op"))

variants = list(elt.fast_variants())
print(f"fast_variants() = {len(variants)} 组：{variants[:4]} …")
print("FASTPATH_OPS =", elt.FASTPATH_OPS)

rows = [(op, v) for op, v in variants if op in elt.FASTPATH_OPS]
print(f"要登记的 (op, variant) = {len(rows)} 组 × 3 dtype = {len(rows) * 3} 个内核")

print("=== 1. 首轮 get_launch_meta（真编译）===")
t0 = time.perf_counter()
first = 0.0
for op, variant in rows:
    for dtype in ("fp32", "fp16", "bf16"):
        t1 = time.perf_counter()
        meta = elt.get_launch_meta(op, dtype, variant)
        first += time.perf_counter() - t1
total_first = time.perf_counter() - t0
print(f"  合计 {total_first * 1000:.0f} ms（其中函数内 {first * 1000:.0f} ms）"
      f"，平均 {total_first * 1000 / (len(rows) * 3):.0f} ms/内核")

print("=== 2. 第二轮（磁盘缓存命中）===")
t0 = time.perf_counter()
for op, variant in rows:
    for dtype in ("fp32", "fp16", "bf16"):
        elt.get_launch_meta(op, dtype, variant)
print(f"  合计 {(time.perf_counter() - t0) * 1000:.0f} ms")

print("=== 3. C++ 侧要用的字段（抽查 add/tensor/fp32 与 mul/tensor/bf16）===")
for op, variant, dtype in (("add", "tensor", "fp32"), ("mul", "tensor", "bf16")):
    m = elt.get_launch_meta(op, dtype, variant)
    spec = elt.fast_op(op, dtype, variant)
    print(f"  {op}/{variant}/{dtype}: n_ptrs={m['n_ptrs']} n_i32={m['n_i32']} n_f32={m['n_f32']} "
          f"alpha_effective={m['alpha_effective']} block={m['block']} warps={m['num_warps']} "
          f"stages={m['num_stages']} cubin={len(m['cubin'])}B args={spec['args']}")
print("FASTOP_COST_DONE")
PYEOF

timeout --foreground -k 5 900 "$PY" -B -u "$W/fastop_cost.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
