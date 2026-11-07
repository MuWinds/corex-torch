set -u
# 目的：把「C++ 到底接管了哪些 schema」变成一条**可复现的判据**，而不是靠时间推算。
# torch28 的 `probes/127` 用了 `torch._C._dispatch_dump_table`，这里我们自己再验一遍：
#   /dev/null:<行号>        = Python 注册（torch.library.Library.impl）
#   corex_shim.cpp:<行号>   = C++ 注册（TORCH_LIBRARY_IMPL）
# 同时量三种实参形态（同形 / 广播 / 标量）的 p50，验证「谁的实现被真正执行」。
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

cat > "$W/dispatch_check.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch
from corex_torch import _device

corex_torch.register()
torch.manual_seed(5)

shim = _device.shim()
print("shim 导出数 =", len([n for n in dir(shim) if not n.startswith("_")]))
for cap in ("elementwise_op_names", "elementwise_kernel_info",
            "elementwise_supports_broadcast"):
    print(f"  hasattr {cap} = {hasattr(shim, cap)}")
if hasattr(shim, "elementwise_op_names"):
    names = sorted(str(n) for n in shim.elementwise_op_names())
    print("  C++ 接管的 schema:", names)

print("=== 1. dispatch 表：谁在提供实现 ===")
dump = torch._C._dispatch_dump_table("aten::add.out")
lines = [ln.strip() for ln in dump.splitlines() if "PrivateUse1" in ln]
print("  aten::add.out     →", lines[0] if lines else "<无>")
dump = torch._C._dispatch_dump_table("aten::add_.Tensor")
lines = [ln.strip() for ln in dump.splitlines() if "PrivateUse1" in ln]
print("  aten::add_.Tensor →", lines[0] if lines else "<无>")
dump = torch._C._dispatch_dump_table("aten::mm")
lines = [ln.strip() for ln in dump.splitlines() if "PrivateUse1" in ln]
print("  aten::mm          →", lines[0] if lines else "<无>")

print("=== 2. 三种形态的 p50（500 次，带空循环基线）===")


def bench(name, fn, iters=500):
    try:
        fn()
        fn()
        torch.corex.synchronize()
    except Exception as exc:                            # noqa: BLE001
        print(f"  {name:40s} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        return None
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    torch.corex.synchronize()
    samples.sort()
    print(f"  {name:40s} p50={samples[len(samples) // 2]:8.1f} us")
    return samples[len(samples) // 2]


a = torch.randn(4, 64).to("corex")
b = torch.randn(4, 64).to("corex")
small = torch.randn(1, 64).to("corex")
out = torch.empty(4, 64, device="corex")

bench("空循环", lambda: None)
bench("a + b（同形，会分配）", lambda: a + b)
bench("a.mul_(2.0)（原地标量）", lambda: a.mul_(2.0))
bench("a * 2.0（标量，会分配）", lambda: a * 2.0)
bench("a * small（广播，会分配）", lambda: a * small)
bench("aten::add.out 直调", lambda: torch.ops.aten.add.out(a, b, alpha=1, out=out))
print("DISPATCH_CHECK_DONE")
PYEOF

timeout --foreground -k 5 900 "$PY" -B -u "$W/dispatch_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
