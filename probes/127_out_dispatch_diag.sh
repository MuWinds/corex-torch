set -u
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
cat > "$W/out_diag.py" <<'PYEOF'
import sys
sys.path.insert(0, "/data/home/2024240384035/coffin-work/base/sandbox/work/ct/src")
import torch, corex_torch
from corex_torch import _runtime as rt
corex_torch.register()
shim = corex_torch._device.shim()
print("elementwise_kernel_info 条数:", len(shim.elementwise_kernel_info()))
for op in ("aten::add.out", "aten::add_.Tensor"):
    print(f"--- {op} ---")
    for line in torch._C._dispatch_dump_table(op).splitlines():
        if "PrivateUse1" in line or "Composite" in line or "CompositeExplicit" in line:
            print("   ", line.strip()[:150])
a = torch.arange(1024, dtype=torch.float32); b = torch.arange(1024, dtype=torch.float32)
ac, bc = a.to("corex"), b.to("corex")
out = torch.empty(1024, device="corex")
import time
def bench(fn, iters=500):
    fn(); rt.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); s.append((time.perf_counter()-t0)*1e6)
    rt.synchronize(); s.sort(); return s[len(s)//2]
print("add.out via torch.add(out=)  p50 =", round(bench(lambda: torch.add(ac, bc, out=out)), 2))
print("add.out via ops.aten        p50 =", round(bench(lambda: torch.ops.aten.add.out(ac, bc, alpha=1, out=out)), 2))
print("add.Tensor (分配+写入)      p50 =", round(bench(lambda: ac + bc), 2))
print("add_.Tensor（就地）         p50 =", round(bench(lambda: torch.ops.aten.add_.Tensor(out, bc, alpha=1)), 2))
PYEOF
timeout --foreground -k 2 120 "$PY" -B -u "$W/out_diag.py"
echo "DIAG_RC=$?"