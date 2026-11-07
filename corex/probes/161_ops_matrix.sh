set -u
# 目的：跑一遍算子矩阵（96 行语义），把「哪些 torch 语义在卡上还不对」一次列清。
# 说明：包与 shim 用上一轮已经覆盖进 torch 树的那份（同一次真机验证的产物）。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cd "$H/work" && "$PY" -u opsmatrix.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | tail -30
echo "T161_DONE"
