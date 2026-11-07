set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
echo "=== 功能 + 耗时 ==="
( cd "$H/work" && "$PY" -u final_verify.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | head -20 )
echo "=== mini decoder ==="
( cd "$H/work" && "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax|ms" | head -6 )
echo "T177_DONE"
