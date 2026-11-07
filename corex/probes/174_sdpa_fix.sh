set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorexf.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
echo "=== sdpa 诊断 ==="
( cd "$H/work" && "$PY" -u sdpa_diag.py 2>&1 | grep -E "^(shape|max_abs|ref scale|manual|dev_manual)" | head -6 )
echo "=== 算子矩阵 ==="
( cd "$H/work" && "$PY" -u opsmatrix2.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | tail -8 )
echo "T174_DONE"
