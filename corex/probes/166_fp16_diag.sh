set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/fp16_diag.py" <<'PYEOF'
import torch

torch.corex.register()
torch.manual_seed(0)
for dt in (torch.float16, torch.bfloat16):
    h = torch.randn(16).to(dt)
    got = (h.to("corex") + h.to("corex"))
    ref = h + h
    print(f"DTYPE {dt} got.dtype={got.dtype} ref.dtype={ref.dtype} shape={tuple(got.shape)}", flush=True)
    g = got.to("cpu").float()
    r = ref.float()
    print("  max_abs_diff", float((g - r).abs().max()), flush=True)
    print("  got[:4]", [round(float(v), 5) for v in g[:4]], flush=True)
    print("  ref[:4]", [round(float(v), 5) for v in r[:4]], flush=True)
PYEOF
cd "$H/work" && "$PY" -u fp16_diag.py 2>&1 | grep -E "^DTYPE|^  " | head -12
echo "T166_DONE"
