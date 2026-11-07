set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
echo "HOST=$(hostname) DATE=$(date -Is)"
cat > "$H/work/tcorex_reg.py" <<'PYEOF'
import traceback
import torch

try:
    torch.corex.register()
    print("MARK_REGISTER_OK", flush=True)
except Exception:
    print("MARK_REGISTER_FAIL", flush=True)
    traceback.print_exc()

try:
    x = torch.zeros(4, 8, device="corex")
    print("MARK_ZEROS", x.device, float(x.sum()), flush=True)
except Exception:
    print("MARK_ZEROS_FAIL", type(x).__name__ if 'x' in dir() else "", flush=True)
    traceback.print_exc()
PYEOF
cd "$H/work" && "$PY" -u tcorex_reg.py > "$H/work/reg.out" 2>&1
echo "=== MARK 行 ==="
grep -nE "MARK_|Error|error:" "$H/work/reg.out" | head -12 | cut -c1-200
echo "=== 首个 traceback 之前 8 行 ==="
sed -n '1,12p' "$H/work/reg.out" | cut -c1-200
echo "T146_DONE"
