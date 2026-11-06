set -u
# 目的：mini decoder 端到端跑通了但数值错（全局相对误差 6.19e-01，argmax 一致率 0.5）。
# 逐阶段比对 CPU 与 corex，找出**第一个**开始发散的算子——不要猜，猜会连着改错三处。
# 顺序：embedding → rms_norm → qkv(mm) → RoPE(含切片写) → scores(bmm) → softmax → bmm(o)
#       → o_proj → 残差 → norm2 → MLP(mm/silu/mm) → 残差 → normf → head
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/stage_diff.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(1234)

VOCAB, DIM, HEADS, T = 32, 64, 4, 8
HD = DIM // HEADS
STAGES = []


def cmp(name, got, ref):
    """记录一个阶段的误差，返回 corex 侧结果供后续阶段继续用。"""
    g = got.cpu().double()
    r = ref.double()
    d = float((g - r).abs().max())
    scale = float(r.abs().max()) or 1.0
    STAGES.append((name, d, d / scale))
    return got


def rope_tables(t, head_dim):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(t).float()[:, None]
    ang = pos * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


w = {k: torch.randn(*s) * 0.1 for k, s in (
    ("wq", (DIM, DIM)), ("wk", (DIM, DIM)), ("wv", (DIM, DIM)), ("wo", (DIM, DIM)),
    ("w1", (DIM, DIM * 2)), ("w2", (DIM * 2, DIM)), ("head", (DIM, VOCAB)))}
w["embed"] = torch.randn(VOCAB, DIM) * 0.3
w["norm1"] = torch.rand(DIM) + 0.5
w["norm2"] = torch.rand(DIM) + 0.5
w["normf"] = torch.rand(DIM) + 0.5
ids = torch.randint(0, VOCAB, (T,))
cos, sin = rope_tables(T, HD)

wd = {k: v.to("corex") for k, v in w.items()}
cos_d, sin_d, ids_d = cos.to("corex"), sin.to("corex"), ids.to("corex")

print("stage-by-stage（每行：CPU 与 corex 同一条式子，单位 max_abs / 全局相对）")
x_c = F.embedding(ids, w["embed"])
x_d = F.embedding(ids_d, wd["embed"])
cmp("1 embedding", x_d, x_c)

h_c = F.rms_norm(x_c, (DIM,), weight=w["norm1"])
h_d = F.rms_norm(x_d, (DIM,), weight=wd["norm1"])
cmp("2 rms_norm", h_d, h_c)

q_c, k_c, v_c = h_c @ w["wq"], h_c @ w["wk"], h_c @ w["wv"]
q_d, k_d, v_d = h_d @ wd["wq"], h_d @ wd["wk"], h_d @ wd["wv"]
cmp("3 q @ wq (mm)", q_d, q_c)
cmp("4 v @ wv (mm)", v_d, v_c)

qr_c = apply_rope(q_c.view(T, HEADS, HD), cos, sin)
qr_d = apply_rope(q_d.view(T, HEADS, HD), cos_d, sin_d)
cmp("5 rope(q) 含切片写", qr_d, qr_c)

# 单看 apply_rope 内部：先看 o1/o2 两步，再看写回
x1_c, x2_c = q_c.view(T, HEADS, HD)[..., 0::2], q_c.view(T, HEADS, HD)[..., 1::2]
x1_d, x2_d = q_d.view(T, HEADS, HD)[..., 0::2], q_d.view(T, HEADS, HD)[..., 1::2]
o1_c = x1_c * cos[:, None, :] - x2_c * sin[:, None, :]
o1_d = x1_d * cos_d[:, None, :] - x2_d * sin_d[:, None, :]
cmp("5a rope o1 = x1*c - x2*s", o1_d, o1_c)

# 用 zeros_like：只写一半时另一半是未初始化内存，empty 会把「未初始化」误判成发散（第一版就踩了）
buf_c = torch.zeros_like(q_c.view(T, HEADS, HD))
buf_d = torch.zeros_like(q_d.view(T, HEADS, HD))
buf_c[..., 0::2] = o1_c
buf_d[..., 0::2] = o1_d
cmp("5b 切片写 dst[...,0::2] = o1", buf_d, buf_c)

scores_c = torch.bmm(qr_c.permute(1, 0, 2).contiguous(),
                     qr_c.permute(1, 0, 2).transpose(1, 2).contiguous()) / (HD ** 0.5)
scores_d = torch.bmm(qr_d.permute(1, 0, 2).contiguous(),
                     qr_d.permute(1, 0, 2).transpose(1, 2).contiguous()) / (HD ** 0.5)
cmp("6 scores = bmm(q, k^T)", scores_d, scores_c)

p_c = torch.softmax(scores_c, dim=-1)
p_d = torch.softmax(scores_d, dim=-1)
cmp("7 softmax", p_d, p_c)

o_c = torch.bmm(p_c, qr_c.permute(1, 0, 2).contiguous())
o_d = torch.bmm(p_d, qr_d.permute(1, 0, 2).contiguous())
cmp("8 bmm(p, v)", o_d, o_c)

flat_c = o_c.permute(1, 0, 2).contiguous().view(T, DIM)
flat_d = o_d.permute(1, 0, 2).contiguous().view(T, DIM)
cmp("9 permute+contiguous", flat_d, flat_c)

proj_c = flat_c @ w["wo"]
proj_d = flat_d @ wd["wo"]
cmp("10 o_proj (mm)", proj_d, proj_c)

y_c = x_c + proj_c
y_d = x_d + proj_d
cmp("11 残差 add", y_d, y_c)

h2_c = F.rms_norm(y_c, (DIM,), weight=w["norm2"])
h2_d = F.rms_norm(y_d, (DIM,), weight=wd["norm2"])
cmp("12 rms_norm #2", h2_d, h2_c)

m_c = F.silu(h2_c @ w["w1"]) @ w["w2"]
m_d = F.silu(h2_d @ wd["w1"]) @ wd["w2"]
cmp("13 MLP (mm+silu+mm)", m_d, m_c)

print("")
for name, d, rel in STAGES:
    flag = "  <-- 从这里开始发散" if rel > 1e-5 else ""
    print(f"  {name:32s} max_abs={d:.3e}  全局相对={rel:.3e}{flag}")

first = next((n for n, _, rel in STAGES if rel > 1e-5), None)
print("FIRST_DIVERGENT", first if first else "NONE")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/stage_diff.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
