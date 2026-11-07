set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorexh.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
export COREX_DEBUG=1
cat > "$H/work/dbg_act.py" <<'PYEOF'
import torch
print("IMPORTED", flush=True)
a = torch.randn(16).to("corex")
b = (a + 1.0) + (a - 1.0)
print("OK", float(b.sum().to("cpu")) > -1e9, flush=True)
PYEOF
( cd "$H/work" && "$PY" -u dbg_act.py 2>&1 | grep -E "corex-triton|IMPORTED|^OK" | head -12 )
echo "T180_DONE"
