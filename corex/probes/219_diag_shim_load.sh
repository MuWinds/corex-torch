set -u
# 目的：把「装完即用」这次失败的真实报错抓出来（上一版探针被 tail 截掉了开头）。
H=$HOME
PY=$H/venvs/fresh28/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "torch 目录=$SITE"
echo "shim=$(stat -c %s "$SITE/corex/_corex_shim.so" 2>/dev/null || echo 缺)"
echo "=== 直接加载 shim 看错误 ==="
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
"$PY" - <<'PYEOF' 2>&1 | head -30
import traceback

import torch

print("TORCH", torch.__version__)
try:
    from torch.corex import _device

    shim = _device.shim()
    print("SHIM_OK", shim)
except Exception:
    traceback.print_exc()
PYEOF
echo "=== ldd ==="
ldd "$SITE/corex/_corex_shim.so" 2>&1 | grep -c "not found" || true
ldd "$SITE/corex/_corex_shim.so" 2>&1 | grep "not found" | head -5
echo "T219_DONE"
