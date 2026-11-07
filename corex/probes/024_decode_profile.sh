set -u
# 目的：decode 是 9.2 ms/步，但把 `probes/023` 的逐算子 p50 按 decode 里的调用来累加只能解释
# 约 3 ms。也就是说要么实际调用的算子比我数的多，要么 decode 里的形状/路径比隔离测量更慢。
# 猜没有意义——这份探针用 cProfile 跑 3 步 decode，按**自身耗时**（tottime）排序，
# 直接看是谁在吃时间。cProfile 有额外开销，但用来做「谁最大」的排序是可靠的。
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

cat > "$W/decode_prof.py" <<'PYEOF'
import cProfile
import io
import pstats
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(2024)

VOCAB, DIM, HEADS, LAYERS, STEPS = 32, 64, 4, 2, 4
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
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def attn_step(h, w, cos, sin, kcache, vcache):
    q = (h @ w["wq"]).view(1, HEADS, HD)
    k = (h @ w["wk"]).view(1, HEADS, HD)
    v = (h @ w["wv"]).view(1, HEADS, HD)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    frame_k = k.squeeze(0).unsqueeze(1)
    frame_v = v.squeeze(0).unsqueeze(1)
    kcache = frame_k if kcache is None else torch.cat((kcache, frame_k), dim=1)
    vcache = frame_v if vcache is None else torch.cat((vcache, frame_v), dim=1)
    qh = q.permute(1, 0, 2)
    scores = torch.bmm(qh, kcache.transpose(1, 2).contiguous()) / (HD ** 0.5)
    p = torch.softmax(scores, dim=-1)
    o = torch.bmm(p, vcache)
    return o.permute(1, 0, 2).contiguous().view(1, DIM), kcache, vcache


def decode_step(tok, pos, w, cos_all, sin_all, caches):
    cos, sin = cos_all[pos:pos + 1], sin_all[pos:pos + 1]
    x = F.embedding(tok, w["embed"])
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
    return x @ w["head"], caches


w = make_weights()
wd = {k: v.to("corex") for k, v in w.items()}
cos_all, sin_all = rope_tables(STEPS + 2, HD)
cos_d, sin_d = cos_all.to("corex"), sin_all.to("corex")
ids = [int(v) for v in torch.randint(0, VOCAB, (STEPS,))]


def run_once():
    caches = [None] * LAYERS
    for step in range(STEPS):
        tok = torch.tensor([ids[step]], device="corex")
        _, caches = decode_step(tok, step, wd, cos_d, sin_d, caches)
    torch.corex.synchronize()


run_once()          # 热身：把内核编译掉
torch.corex.synchronize()
t0 = time.perf_counter()
run_once()
print(f"未 profile 的单次 {STEPS} 步耗时 = {1000 * (time.perf_counter() - t0):.1f} ms "
      f"→ {1000 * (time.perf_counter() - t0) / STEPS:.2f} ms/步")

prof = cProfile.Profile()
prof.enable()
run_once()
prof.disable()
stream = io.StringIO()
stats = pstats.Stats(prof, stream=stream)
stats.sort_stats("tottime")
stats.print_stats(28)
text = stream.getvalue()
# 只保留表头和排序后的正文，去掉 cProfile 的样板
lines = [ln for ln in text.splitlines() if ln.strip()]
print("=== cProfile（按 tottime 排序，前 28 项）===")
for ln in lines[-34:]:
    print(ln[:150])
print("PROF_DONE")
PYEOF

timeout --foreground -k 5 900 "$PY" -B -u "$W/decode_prof.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
