set -u
# 目的：把「算子清单」换成「一条真实前向」——用一个 mini decoder block 端到端对齐 CPU。
# 为什么这条比逐算子表更重要：算子是「我猜推理需要它」，前向是「推理真的用到了它」。
# 一条能跑通的前向会立刻暴露「还缺什么」，而且暴露的是**组合**层面的缺口（视图 + 归约 + 矩阵乘
# + 逐元素的顺序与 stride 假设），单算子表永远测不出来。
#
# 结构（尺寸取小，方便 CPU 参照与逐元素比对；形状比例按真实 decoder）：
#   token id → embedding(index_select) → RmsNorm → q/k/v(mm) → RoPE → 多头注意力(bmm+softmax+bmm)
#   → o_proj → 残差 → RmsNorm → MLP(mm+silu+mm) → 残差 → 最终 RmsNorm → logits(mm)
# 判据：与 CPU 同结构的输出比**全局相对误差**（max|diff| / max|ref|），并且逐算子链路上不出现
#       NotImplementedError。
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

cat > "$W/mini_decoder.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(1234)

VOCAB, DIM, HEADS, T, LAYERS = 32, 64, 4, 8, 2
HD = DIM // HEADS


def make_weights():
    """所有参数都在 CPU 上随机生成，再整体搬到卡上（卡上没有 RNG）。"""
    w = {
        "embed": torch.randn(VOCAB, DIM) * 0.3,
        "wq": torch.randn(DIM, DIM) * 0.1,
        "wk": torch.randn(DIM, DIM) * 0.1,
        "wv": torch.randn(DIM, DIM) * 0.1,
        "wo": torch.randn(DIM, DIM) * 0.1,
        "norm1": torch.rand(DIM) + 0.5,
        "norm2": torch.rand(DIM) + 0.5,
        "normf": torch.rand(DIM) + 0.5,
        "w1": torch.randn(DIM, DIM * 2) * 0.1,
        "w2": torch.randn(DIM * 2, DIM) * 0.1,
        "head": torch.randn(DIM, VOCAB) * 0.1,
    }
    return w


def rope_tables(t, head_dim):
    """RoPE 的 cos/sin，按 (t, head_dim/2)——注意是**每头的维度**，不是模型维度。"""
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(t).float()[:, None]
    ang = pos * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: [T, HEADS, HD]；对最后一维按相邻两两配对旋转。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]        # [T, HEADS, HD/2]
    c, s = cos[:, None, :], sin[:, None, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    # 交错拼回：先分配输出再把两半写进去（避免依赖 cat 的实现细节）
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


def block(x, w, cos, sin, device):
    """一个 decoder block。x: [T, DIM]。"""
    h = F.rms_norm(x, (DIM,), weight=w["norm1"])
    q = h @ w["wq"]
    k = h @ w["wk"]
    v = h @ w["wv"]
    # [T, DIM] -> [T, HEADS, HD]
    q = q.view(T, HEADS, HD)
    k = k.view(T, HEADS, HD)
    v = v.view(T, HEADS, HD)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    # 注意力：按头做 bmm，[HEADS, T, HD] x [HEADS, HD, T]
    qh = q.permute(1, 0, 2).contiguous()
    kh = k.permute(1, 0, 2).contiguous()
    vh = v.permute(1, 0, 2).contiguous()
    scores = torch.bmm(qh, kh.transpose(1, 2).contiguous()) / (HD ** 0.5)
    p = torch.softmax(scores, dim=-1)
    oh = torch.bmm(p, vh)                        # [HEADS, T, HD]
    o = oh.permute(1, 0, 2).contiguous().view(T, DIM)
    x = x + o @ w["wo"]
    h = F.rms_norm(x, (DIM,), weight=w["norm2"])
    x = x + (F.silu(h @ w["w1"]) @ w["w2"])
    return x


def forward(ids, w, cos, sin, device):
    x = F.embedding(ids, w["embed"])             # index_select 通道
    for _ in range(LAYERS):
        x = block(x, w, cos, sin, device)
    x = F.rms_norm(x, (DIM,), weight=w["normf"])
    return x @ w["head"]                         # [T, VOCAB]


def to_dev(w):
    return {k: v.to("corex") for k, v in w.items()}


ids = torch.randint(0, VOCAB, (T,))
w_cpu = make_weights()
cos, sin = rope_tables(T, HD)

print("=== 1. CPU 参照 ===")
t0 = time.perf_counter()
ref = forward(ids, w_cpu, cos, sin, "cpu")
print(f"  {1000 * (time.perf_counter() - t0):.1f} ms  shape={tuple(ref.shape)}")

print("=== 2. corex 上同一条前向 ===")
w_dev = to_dev(w_cpu)
cos_d, sin_d = cos.to("corex"), sin.to("corex")
torch.corex.synchronize()
t0 = time.perf_counter()
got = forward(ids.to("corex"), w_dev, cos_d, sin_d, "corex")
torch.corex.synchronize()
elapsed = 1000 * (time.perf_counter() - t0)
print(f"  {elapsed:.1f} ms（含首次内核编译）")

print("=== 3. 比对 ===")
got_cpu = got.cpu().double()
ref_d = ref.double()
max_abs = float((got_cpu - ref_d).abs().max())
scale = float(ref_d.abs().max())
rel = max_abs / scale if scale else max_abs
print(f"  max_abs = {max_abs:.3e}   max|ref| = {scale:.3e}   全局相对误差 = {rel:.3e}")
print(f"  logits argmax 一致率 = "
      f"{float((got_cpu.argmax(-1) == ref_d.argmax(-1)).float().mean()):.3f}")

print("=== 4. 第二次（热）===")
t0 = time.perf_counter()
got2 = forward(ids.to("corex"), w_dev, cos_d, sin_d, "corex")
torch.corex.synchronize()
print(f"  {1000 * (time.perf_counter() - t0):.1f} ms")

verdict = rel <= 1e-5
print("MINI_DECODER_VERDICT", "PASS" if verdict else "FAIL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/mini_decoder.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
