set -u
# 目的：装本次改好的 torch/corex 覆盖到已装 torch，然后跑功能验证（register + 显存分配 + 加法）。
H=$HOME
PY=$H/venvs/corex28/bin/python
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
rm -rf "$SITE/corex"; mkdir -p "$SITE/corex"
tar xzf "$H/work/tcorex2.tgz" -C "$SITE/corex" --strip-components=2 || exit 1
SO=$(find "$H/coffin-work" -name "_corex_shim.so" 2>/dev/null | head -1)
[ -n "$SO" ] && cp -f "$SO" "$SITE/corex/_corex_shim.so"
echo "shim=${SO:-无}  files=$(ls "$SITE/corex" | wc -l)"
cat > "$H/work/t3.py" <<'PYEOF'
import traceback
import torch
try:
    torch.corex.register()
    print("MARK_REG_OK", torch.corex.is_available(), torch.corex.device_count(), flush=True)
except Exception:
    print("MARK_REG_FAIL", flush=True); traceback.print_exc()
try:
    x = torch.zeros(4, 8, device="corex")
    a = torch.arange(8.0)
    b = a.to("corex") + 1.0
    print("MARK_ZEROS", x.device, float(x.sum()), torch.equal(b.to("cpu"), a + 1.0), flush=True)
except Exception:
    print("MARK_ZEROS_FAIL", flush=True); traceback.print_exc()
PYEOF
cd "$H/work" && "$PY" -u t3.py > "$H/work/t3.out" 2>&1
grep -E "MARK_|Error|error:" "$H/work/t3.out" | head -8 | cut -c1-190
echo "T147_DONE"
