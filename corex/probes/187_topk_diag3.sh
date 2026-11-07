set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorexk.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
export COREX_DEBUG=1
( cd "$H/work" && "$PY" -u topk_diag.py 2>&1 | grep -E "^(A|B|C|D) |corex\]|match" | head -8 )
echo "T186_DONE"
