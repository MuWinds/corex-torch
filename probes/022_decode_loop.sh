set -u
# 目的：把「一条前向」升级成「**逐步生成**」——这是 coffin 真正要用的形态，也是唯一能验
# KV cache 这条路的测试。018 是无 cache 的整段前向，它证明不了：
#   1. 每步只喂 1 个 token 时，注意力在 `[HEADS, 1, HD] @ [HEADS, HD, L]` 这种非方阵上对不对；
#   2. KV cache 用 `torch.cat` 增长后，历史位置的 key/value 是否与「重新算一遍」一致；
#   3. 逐 token 的 logits 会不会随步数漂移（误差累积）。
#
# 做法：同一条 decode 循环在 CPU 与 corex 上各跑一遍，**每一步**都比 logits。
# 判据：全局相对误差 ≤1e-5，且每步的 argmax 与 CPU 一致。
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

cat > "$W/decode_loop.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(2024)

VOCAB, DIM, HEADS, LAYERS, STEPS = 32, 64, 4, 2, 6
HD = DIM // HEADS


def make_weights():
    w = {}
    for name in ("wq", "wk", "wv", "wo"):
        w[name] = torch.randn(DIM, DIM) * 0.12
    w["w1"] = torch.randn(DIM, DIM * 2) * 0.12
    w["w2"] = torch.randn(DIM * 2, DIM) * 0.12
    w["head"] = torch.randn(DIM, VOCAB) * 0.12
    w["embed"] = torch.randn(VOCAB, DIM) * 0.3
    for name in ("norm1", "norm2", "normf"):
        w[name] = torch.rand(DIM) + 0.5
    return w


def rope_tables(t, head_dim):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(t).float()[:, None]
    ang = pos * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: [1, HEADS, HD]；cos/sin: [1, HD/2]。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def attn_step(h, w, cos, sin, kcache, vcache):
    """单 token 的注意力：q 是 1 个位置，k/v 是 cache 里的全部历史。"""
    q = (h @ w["wq"]).view(1, HEADS, HD)
    k = (h @ w["wk"]).view(1, HEADS, HD)
    v = (h @ w["wv"]).view(1, HEADS, HD)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    # cache 形状 [HEADS, L, HD]：新的一帧 append 到 L 维。
    #
    # 这里不能用 `permute(1,0,2).contiguous()`：`[1,HEADS,HD] -> [HEADS,1,HD]` 的形状里有一维是 1，
    # 而 torch 的连续性判定**忽略长度为 1 的维**，于是 `.contiguous()` 是空操作，stride 仍是
    # (16, 64, 1)；我们的 `cat` 按「dim 之后每个维的 stride 必须等于尾积」检查，就会拒。
    # 用 squeeze/unsqueeze 得到的 stride 是 (16, 16, 1)，两边都认。
    frame_k = k.squeeze(0).unsqueeze(1)
    frame_v = v.squeeze(0).unsqueeze(1)
    kcache = frame_k if kcache is None else torch.cat((kcache, frame_k), dim=1)
    vcache = frame_v if vcache is None else torch.cat((vcache, frame_v), dim=1)
    qh = q.permute(1, 0, 2)                                  # [HEADS, 1, HD]
    scores = torch.bmm(qh, kcache.transpose(1, 2).contiguous()) / (HD ** 0.5)
    p = torch.softmax(scores, dim=-1)
    o = torch.bmm(p, vcache)                                 # [HEADS, 1, HD]
    return o.permute(1, 0, 2).contiguous().view(1, DIM), kcache, vcache


def decode_step(tok, pos, w, cos_all, sin_all, caches, device):
    """一步 decode：tok 是 (1,) 的 id 张量，pos 是当前绝对位置。"""
    cos, sin = cos_all[pos:pos + 1], sin_all[pos:pos + 1]
    x = F.embedding(tok, w["embed"])                         # [1, DIM]
    for layer in range(LAYERS):
        caches[layer] = caches[layer] or (None, None)
        kc, vc = caches[layer]
        h = F.rms_norm(x, (DIM,), weight=w["norm1"])
        o, kc, vc = attn_step(h, w, cos, sin, kc, vc)
        x = x + o @ w["wo"]
        h = F.rms_norm(x, (DIM,), weight=w["norm2"])
        x = x + (F.silu(h @ w["w1"]) @ w["w2"])
        caches[layer] = (kc, vc)
    x = F.rms_norm(x, (DIM,), weight=w["normf"])
    return x @ w["head"], caches                              # [1, VOCAB]


ids = [int(v) for v in torch.randint(0, VOCAB, (STEPS,))]
w = make_weights()
cos_all, sin_all = rope_tables(STEPS, HD)


def run(device):
    """跑同一条 decode 循环，返回每一步的 logits（都搬回 CPU）。"""
    wd = {k: v.to(device) for k, v in w.items()}
    cos_d, sin_d = cos_all.to(device), sin_all.to(device)
    caches = [None] * LAYERS
    logits = []
    for step in range(STEPS):
        tok = torch.tensor([ids[step]], device=device)
        lg, caches = decode_step(tok, step, wd, cos_d, sin_d, caches, device)
        logits.append(lg)
    if device != "cpu":
        torch.corex.synchronize()
    return logits


print(f"=== 1. CPU 逐步 decode（{STEPS} 步）===")
t0 = time.perf_counter()
ref = run("cpu")
print(f"  {1000 * (time.perf_counter() - t0):.1f} ms")

print("=== 2. corex 逐步 decode ===")
t0 = time.perf_counter()
got = run("corex")
print(f"  {1000 * (time.perf_counter() - t0):.1f} ms（含首次内核编译）")

t0 = time.perf_counter()
run("corex")
print(f"  热跑 {1000 * (time.perf_counter() - t0):.1f} ms / {STEPS} 步 "
      f"= {1000 * (time.perf_counter() - t0) / STEPS:.1f} ms/步")

print("=== 3. 逐 token 比对（注意 KV cache 会放大早期误差，这条必须逐项看）===")
worst = 0.0
argmax_ok = 0
for step in range(STEPS):
    g = got[step].cpu().double()
    r = ref[step].double()
    d = float((g - r).abs().max())
    scale = float(r.abs().max()) or 1.0
    rel = d / scale
    worst = max(worst, rel)
    same = int(g.argmax(-1)[0]) == int(r.argmax(-1)[0])
    argmax_ok += 1 if same else 0
    print(f"  step {step}: 全局相对={rel:.3e}  argmax一致={same}")

print(f"最差全局相对 = {worst:.3e}   argmax 一致 {argmax_ok}/{STEPS}")

print("=== 4. KV cache 一致性：把 cache 里历史部分与「重算」对照 ===")
# 用最后一步的 cache 验一件事：历史位置的 k 不应随步数变化（RoPE 是按绝对位置加的）。
# 这里只做「同一份 cache 在新旧两步之间是否数值稳定」的弱检查：重跑一次，cache 应逐字节相同。
def cache_snapshot(device):
    wd = {k: v.to(device) for k, v in w.items()}
    cos_d, sin_d = cos_all.to(device), sin_all.to(device)
    caches = [None] * LAYERS
    for step in range(STEPS):
        tok = torch.tensor([ids[step]], device=device)
        _, caches = decode_step(tok, step, wd, cos_d, sin_d, caches, device)
    if device != "cpu":
        torch.corex.synchronize()
    return caches


c1 = cache_snapshot("corex")
c2 = cache_snapshot("corex")
stable = all(torch.equal(a[0].cpu(), b[0].cpu()) and torch.equal(a[1].cpu(), b[1].cpu())
             for a, b in zip(c1, c2))
print(f"  两次独立 decode 的 KV cache 逐字节相同 = {stable}")

verdict = worst <= 1e-5 and argmax_ok == STEPS and stable
print("DECODE_LOOP_VERDICT", "PASS" if verdict else "FAIL")
PYEOF

timeout --foreground -k 5 900 "$PY" -B -u "$W/decode_loop.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
