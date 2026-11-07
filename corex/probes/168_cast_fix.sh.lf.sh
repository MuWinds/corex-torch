set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorexa.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
echo "=== cast 诊断 ==="
( cd "$H/work" && "$PY" -u cast_diag.py 2>&1 | grep -E "^DTYPE|^  " | head -8 )
echo "=== 算子矩阵 ==="
( cd "$H/work" && "$PY" -u opsmatrix2.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | tail -10 )
echo "T168_DONE"
