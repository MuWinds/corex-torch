set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/why_plugin.py" <<'PYEOF'
import traceback

from torch.corex import _triton as T

print("FORK?", T._is_fork_backend(), flush=True)
try:
    from torch.corex import _bf16  # noqa: F401

    print("BF16_OK", flush=True)
except Exception:
    print("BF16_FAIL", flush=True)
    traceback.print_exc()
try:
    T.activate()
    from triton.backends import backends

    print("ACTIVE", sorted(backends), flush=True)
except Exception:
    print("ACTIVE_FAIL", flush=True)
    traceback.print_exc()
PYEOF
cd "$H/work" && "$PY" -u why_plugin.py 2>&1 | grep -E "^(FORK|BF16|ACTIVE)|Error|error" | head -8 | cut -c1-170
echo "T158_DONE"
