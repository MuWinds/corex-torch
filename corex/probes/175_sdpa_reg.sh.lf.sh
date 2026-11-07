set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/sdpa_reg.py" <<'PYEOF'
import traceback

import torch

torch.corex.register()
from torch.corex import _aten_kernels as K

lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
for name in ("scaled_dot_product_attention", "_scaled_dot_product_attention_math"):
    try:
        lib.impl(name, K._sdpa_impl, "PrivateUse1")
        print("REG_OK", name, flush=True)
    except Exception:
        print("REG_FAIL", name, flush=True)
        traceback.print_exc()
# 直接调一次
q = torch.randn(1, 2, 4, 8)
k = torch.randn(1, 2, 4, 8)
v = torch.randn(1, 2, 4, 8)
try:
    out = torch.nn.functional.scaled_dot_product_attention(q.to("corex"), k.to("corex"), v.to("corex"))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print("SDPA_DIFF", float((out.to("cpu") - ref).abs().max()), flush=True)
except Exception:
    print("SDPA_FAIL", flush=True)
    traceback.print_exc()
PYEOF
cd "$H/work" && "$PY" -u sdpa_reg.py 2>&1 | grep -E "^(REG_|SDPA_)|Error|error" | head -10 | cut -c1-170
echo "T175_DONE"
