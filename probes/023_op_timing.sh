set -u
# 目的：`probes/022` 的 decode 是 17.1 ms/步，而发射地板只有 10.8 µs/次——差两个数量级，
# 说明瓶颈不在内核。这份探针用「差分法」把 decode 里出现的算子逐个计时（每个跑 N 次，
# 循环外同步一次），得到一张 µs/算子的表，用来定位到底是谁贵。
# 差分法而不是 profiler：这张卡上 torch.profiler 不可靠（且我们不想依赖它的实现细节）。
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

cat > "$W/op_timing.py" <<'PYEOF'
import sys
import time

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(3)

DIM, HEADS, HD, L, VOCAB = 64, 4, 16, 6, 32
ITERS = 50


def dev(t):
    return t.to("corex")


def bench(name, fn, iters=ITERS):
    """跑 iters 次，循环外同步一次；先热身两次把首次编译排除。"""
    fn()
    fn()
    torch.corex.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.corex.synchronize()
    us = 1e6 * (time.perf_counter() - t0) / iters
    print(f"  {name:46s} {us:9.1f} us")
    return us


x1 = dev(torch.randn(1, DIM))
w = dev(torch.randn(DIM, DIM))
w2 = dev(torch.randn(DIM, DIM * 2))
ids = dev(torch.tensor([3]))
emb = dev(torch.randn(VOCAB, DIM))
gamma = dev(torch.rand(DIM) + 0.5)
q = dev(torch.randn(HEADS, 1, HD))
k = dev(torch.randn(HEADS, L, HD))
v = dev(torch.randn(HEADS, L, HD))
scores = dev(torch.randn(HEADS, 1, L))
p = dev(torch.randn(HEADS, 1, L))
xbig = dev(torch.randn(1, DIM))
scale = 0.25
rows = dev(torch.randn(4096))


def timed(name, fn, iters=ITERS):
    try:
        return bench(name, fn, iters)
    except Exception as exc:                            # noqa: BLE001
        print(f"  {name:46s} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        return None


print("=== A. decode 里出现的算子（µs/次，含 Python 派发）===")
timed("empty(1,64) 分配", lambda: torch.empty(1, DIM, device="corex"))
timed("embedding(1 id)", lambda: F.embedding(ids, emb))
timed("rms_norm [1,64]", lambda: F.rms_norm(x1, (DIM,), weight=gamma))
timed("mm [1,64]x[64,64]", lambda: x1 @ w)
timed("mm [1,64]x[64,128]", lambda: x1 @ w2)
timed("bmm [4,1,16]x[4,16,6]", lambda: torch.bmm(q, k.transpose(1, 2).contiguous()))
timed("softmax [4,1,6] dim=-1", lambda: torch.softmax(scores, dim=-1))
timed("bmm [4,1,6]x[4,6,16]", lambda: torch.bmm(p, v))
timed("cat [4,6,16]+[4,1,16] dim=1", lambda: torch.cat((k, dev(torch.randn(HEADS, 1, HD))), dim=1))
timed("permute+squeeze/unsqueeze（视图）", lambda: q.squeeze(-2).unsqueeze(1))
timed("add [1,64]+[1,64]", lambda: x1 + x1)
timed("silu [1,128]", lambda: F.silu(dev(torch.randn(1, DIM * 2))))
timed("mul 标量 [1,64]", lambda: x1 * scale)
timed("mul 广播 [4,1,16]*[1,1,16]-like", lambda: dev(torch.randn(HEADS, 1, HD)) * dev(torch.randn(1, 1, HD)))
timed("slice+contiguous [1,64][...,0::2]", lambda: x1[..., 0::2].contiguous())
timed("copy_ 整块 [1,64]", lambda: torch.empty(1, DIM, device="corex").copy_(x1))
timed("sum() 全归约 4096", lambda: rows.sum())
timed("var(dim=-1) 4096x1", lambda: dev(torch.randn(64, 64)).var(dim=-1))

print("=== B. 纯 Python 地板（不上卡）===")
t0 = time.perf_counter()
for _ in range(ITERS * 20):
    pass
print(f"  {'空循环':46s} {1e6 * (time.perf_counter() - t0) / (ITERS * 20):9.3f} us")
print("=== C. 分层拆解：同一个 add 走三层各多贵 ===")
from corex_torch import _ops_elementwise, _triton as _t

N = 64
pa = torch.empty(N, device="corex")
pb = torch.empty(N, device="corex")
pc = torch.empty(N, device="corex")


def kernel_only():
    _ops_elementwise.add(pa.data_ptr(), pb.data_ptr(), pc.data_ptr(), N, "fp32")


try:
    k_obj = _ops_elementwise._get_kernel("add", "fp32", None, 1024, 1, 1)
    print("  已取到 elementwise 内核对象:", type(k_obj).__name__)
except Exception as exc:                                # noqa: BLE001
    k_obj = None
    print("  取内核对象失败:", type(exc).__name__, str(exc)[:80])

timed("kernel 直调（含我们自己那层 launch）", kernel_only, 200)
if k_obj is not None:
    def launch_only():
        _t.launch(k_obj, (1,), pa.data_ptr(), pb.data_ptr(), pc.data_ptr(), N)

    timed("_triton.launch 裸调（同一内核）", launch_only, 200)
timed("aten add（x1 + x1）", lambda: x1 + x1, 200)
timed("aten mul 标量（x1 * 2.0）", lambda: x1 * 2.0, 200)

print("OP_TIMING_DONE")

PYEOF

timeout --foreground -k 5 900 "$PY" -B -u "$W/op_timing.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
