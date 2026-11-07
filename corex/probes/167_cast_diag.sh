set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/cast_diag.py" <<'PYEOF'
import torch

torch.corex.register()
torch.manual_seed(0)
for dt in (torch.float16, torch.bfloat16):
    h = torch.randn(8).to(dt)
    on_dev = h.to("corex")
    cast_dev = on_dev.float()                 # 设备上做 fp16 -> fp32
    ref = h.float()                           # CPU 上做同样转换
    print(f"DTYPE {dt}: dev_cast dtype={cast_dev.dtype} device={cast_dev.device}", flush=True)
    g = cast_dev.to("cpu")
    print("  max_abs_diff", float((g - ref).abs().max()), flush=True)
    print("  got", [round(float(v), 5) for v in g[:4]], flush=True)
    print("  ref", [round(float(v), 5) for v in ref[:4]], flush=True)
    # 反向：fp32 -> fp16
    back = cast_dev.to(dt)
    print("  back dtype", back.dtype, "diff", float((back.to("cpu").float() - h.float()).abs().max()), flush=True)
PYEOF
cd "$H/work" && "$PY" -u cast_diag.py 2>&1 | grep -E "^DTYPE|^  " | head -14
echo "T167_DONE"
