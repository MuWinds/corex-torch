set -u
# 目的：验证「通用一元数学内核 + .out 注册」这一批补完后，算子矩阵能到多少。
# 依赖：远端 ~/work/shim/{corex_shim.cpp,build_shim.py}（含广播物化的修复）与 venv 里的 ninja。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
export PATH=$H/venvs/corex28/bin:$PATH
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 1. 重编 shim（含广播 pitch 修复）==="
"$PY" "$H/work/shim/build_shim.py" --src "$H/work/shim/corex_shim.cpp" \
      --out "$SITE/corex/_corex_shim.so" 2>&1 | tail -3

echo "=== 2. 覆盖 python 包 ==="
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorex6.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/"
rm -rf "$SITE/corex_tmp"

export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

echo "=== 3. 功能矩阵（22 条）==="
( cd "$H/work" && "$PY" -u matrix.py 2>&1 | grep -E "^(OK|FAIL|SUMMARY)" | tail -4 )

echo "=== 4. 算子矩阵（65 条）==="
( cd "$H/work" && "$PY" -u opsmatrix.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | tail -26 )
echo "T162_DONE"
