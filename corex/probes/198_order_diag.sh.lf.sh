set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
export COREX_DEBUG=1
cat > "$H/work/order.py" <<'PYEOF'
import torch  # 这里就会触发 corex 注册与热路径内核登记
print("IMPORT_DONE", flush=True)
PYEOF
cd "$H/work" && "$PY" -u order.py 2>&1 | grep -nE "corex-triton|UserWarning|IMPORT_DONE" | head -20 | cut -c1-150
echo "T198_DONE"
