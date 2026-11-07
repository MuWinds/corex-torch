set -u
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
cat > "$H/work/sdpa_diag.py" <<'PYEOF'
import torch
import torch.nn.functional as F

torch.corex.register()
torch.manual_seed(0)
q = torch.randn(2, 4, 8, 16)
k = torch.randn(2, 4, 8, 16)
v = torch.randn(2, 4, 8, 16)
ref = F.scaled_dot_product_attention(q, k, v)
got = F.scaled_dot_product_attention(q.to("corex"), k.to("corex"), v.to("corex")).to("cpu")
print("shape", tuple(got.shape), "dtype", got.dtype, flush=True)
d = (got - ref).abs()
print("max_abs_diff", float(d.max()), "mean_abs_diff", float(d.mean()), flush=True)
print("ref scale", float(ref.abs().max()), flush=True)
print("got[0,0,0,:4]", [round(float(x), 4) for x in got[0, 0, 0, :4]], flush=True)
print("ref[0,0,0,:4]", [round(float(x), 4) for x in ref[0, 0, 0, :4]], flush=True)
# 手动复算：softmax(q k^T / sqrt(d)) v
s = (q @ k.transpose(-2, -1)) / (16 ** 0.5)
p = s.softmax(-1)
man = p @ v
print("manual_vs_ref", float((man - ref).abs().max()), flush=True)
# 在卡上手动复算，看是哪一步开始偏
sd = (q.to("corex") @ k.to("corex").transpose(-2, -1)) / (16 ** 0.5)
pd = sd.softmax(-1)
md = (pd @ v.to("corex")).to("cpu")
print("dev_manual_vs_ref", float((md - man).abs().max()), flush=True)
PYEOF
cd "$H/work" && "$PY" -u sdpa_diag.py 2>&1 | grep -E "^(shape|max_abs|ref scale|got\[|ref\[|manual|dev_manual)" | head -10
echo "T173_DONE"
