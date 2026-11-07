set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
echo "HOST=$(hostname) DATE=$(date -Is)"
cat > "$H/work/tcorex_reg.py" <<'PYEOF'
import traceback
import torch

print("before", torch.corex)
try:
    torch.corex.register()
    print("REGISTER_OK")
except Exception:
    print("REGISTER_FAIL")
    traceback.print_exc()

try:
    x = torch.zeros(4, 8, device="corex")
    print("ZEROS", x.device, float(x.sum()))
except Exception:
    print("ZEROS_FAIL")
    traceback.print_exc()
PYEOF
cd "$H/work" && "$PY" -u tcorex_reg.py 2>&1 | head -40
echo "T145_DONE"
