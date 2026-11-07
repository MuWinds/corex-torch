set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/ln_bisect.py" <<'PYEOF'
import torch
import torch.corex

torch.corex.register()
F = torch.nn.functional

x = torch.randn(4, 32)
w = torch.randn(32)
b = torch.randn(32)


def case(name, fn):
    try:
        out = fn()
        ref = None
        print(f"OK   {name} sum={float(out.sum().to('cpu')):.4f}", flush=True)
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__} {str(exc)[:110]}", flush=True)


xd = x.to("corex")
wd = w.to("corex")
bd = b.to("corex")
case("no_affine", lambda: F.layer_norm(xd, (32,)))
case("weight_only", lambda: F.layer_norm(xd, (32,), wd, None))
case("bias_only", lambda: F.layer_norm(xd, (32,), None, bd))
case("both", lambda: F.layer_norm(xd, (32,), wd, bd))
case("weight_cpu", lambda: F.layer_norm(xd, (32,), w, b))
case("plain_copy", lambda: torch.zeros(4, 32, device="corex").copy_(torch.zeros(4, 32)))
case("expand_copy", lambda: torch.zeros(4, 32, device="corex").copy_(torch.zeros(32).expand(4, 32)))
case("bcast_add", lambda: xd + wd)
case("contig_view", lambda: xd.t().contiguous())
PYEOF
( cd "$H/work" && "$PY" -u ln_bisect.py 2>&1 | grep -E "^(OK|FAIL)" | head -12 )
echo "T153_DONE"
