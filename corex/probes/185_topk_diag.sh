set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/topk_diag.py" <<'PYEOF'
import torch

torch.corex.register()
x = torch.randn(8, 16)
d = x.to("corex")
print("A device ready", flush=True)
v, i = d.topk(4, -1)
print("B topk ok", tuple(v.shape), tuple(i.shape), v.dtype, i.dtype, flush=True)
print("   values match", torch.allclose(v.to("cpu"), x.topk(4, -1).values), flush=True)
print("   indices match", torch.equal(i.to("cpu"), x.topk(4, -1).indices), flush=True)
r = d.sort(-1)
print("C sort ok", torch.allclose(r.values.to("cpu"), x.sort(-1).values), flush=True)
a = d.argsort(-1)
print("D argsort ok", torch.equal(a.to("cpu"), x.argsort(-1)), flush=True)
PYEOF
cd "$H/work" && "$PY" -u topk_diag.py 2>&1 | grep -E "^[A-D] |match|ok" | head -8
echo "RC=$?"
echo "T185_DONE"
