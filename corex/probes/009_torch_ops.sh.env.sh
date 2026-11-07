export COREX_FAST_MM=1
set -u
# 目的：在 **torch 层**验收「推理要用到的那批算子」，而不是在内核层再验一遍。
# 内核层（probes/180-182）已经证明每个 kernel 与 CPU 参照对得上；这一层要回答的是另一件事：
# aten 的分发、形状/stride 检查、out 张量的形状对账、dtype 转换有没有接错。
#
# 用例形状按推理路径的形态挑：mm/bmm（投影与 batch 头）、softmax（注意力权重）、
# sum/mean（规约）、index_select（embedding 取行），最后拼一条「注意力小块」看能不能连起来。
#
# 判据分三档，都写在输出里：
#   - 逐字节相等：纯搬运/无累加顺序依赖的（index_select、max）
#   - 相对误差：有 fp32 累加顺序差异的（mm/bmm/sum/softmax），门槛 1e-5（softmax 1e-6 绝对值）
#   - 不一致就 FAIL，不写「误差够小」这种含糊话
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/ops_check.py" <<'PYEOF'
import sys
import time
import traceback

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch

corex_torch.register()
torch.manual_seed(0)

FAILED = []
PASSED = []


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """最大相对误差（分母取 max(|b|, 1e-6)，避免接近 0 时炸成 inf）。"""
    diff = (a.cpu().double() - b.cpu().double()).abs()
    den = b.cpu().double().abs().clamp_min(1e-6)
    return float((diff / den).max())


def rel_global(a: torch.Tensor, b: torch.Tensor) -> float:
    """全局相对误差 = max|diff| / max|ref|。

    为什么需要第二种口径：逐元素相对误差在「参照值接近 0」的元素上会被人为放大——
    注意力小块的输出有正有负、量级 1e-3，某个元素恰好接近 0 时，1e-7 的绝对误差就能算出
    1e-4 的相对误差，看着像错其实是 fp32 累加的正常结果。多算子串联（两次矩阵乘 + softmax +
    再一次矩阵乘）必须用全局口径，否则永远判不过。
    """
    diff = (a.cpu().double() - b.cpu().double()).abs().max()
    scale = b.cpu().double().abs().max().clamp_min(1e-12)
    return float(diff / scale)


def check(name, fn, tol=1e-5, byte_equal=False, metric="per_element"):
    try:
        got, ref = fn()
        if byte_equal:
            same = torch.equal(got.cpu(), ref)
            value = 0.0 if same else float("inf")
            ok = same
            extra = "byte-equal" if same else "字节不等"
        else:
            value = rel_global(got, ref) if metric == "global" else rel(got, ref)
            ok = value <= tol
            label = "max_rel(全局)" if metric == "global" else "max_rel"
            extra = f"{label}={value:.3e}（门槛 {tol:.0e}）"
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:38s} {extra}")
        (PASSED if ok else FAILED).append(name)
    except Exception as exc:                            # noqa: BLE001
        head = str(exc).splitlines()[0][:130]
        print(f"  [FAIL] {name:38s} {type(exc).__name__}: {head}")
        FAILED.append(f"{name} -> {type(exc).__name__}: {head}")


print("=== 1. 矩阵乘 ===")

def _mm():
    a = torch.randn(64, 128) * 0.5
    b = torch.randn(128, 256) * 0.5
    return (a.to("corex") @ b.to("corex")), (a @ b)

check("mm 64x128 @ 128x256", _mm)


def _bmm():
    a = torch.randn(4, 16, 32) * 0.5
    b = torch.randn(4, 32, 24) * 0.5
    return torch.bmm(a.to("corex"), b.to("corex")), torch.bmm(a, b)

check("bmm 4x16x32 @ 4x32x24", _bmm)

print("=== 2. softmax / 规约 ===")

def _softmax():
    x = torch.randn(32, 512) * 2.0
    return torch.softmax(x.to("corex"), dim=-1), torch.softmax(x, dim=-1)

check("softmax 32x512 dim=-1", _softmax, tol=1e-6)


def _softmax_3d():
    x = torch.randn(2, 8, 256) * 2.0
    return torch.softmax(x.to("corex"), dim=-1), torch.softmax(x, dim=-1)

check("softmax 2x8x256 dim=-1", _softmax_3d, tol=1e-6)


def _sum_all():
    x = torch.randn(4096) * 0.5
    return x.to("corex").sum(), x.sum()

check("sum() 全归约 4096", _sum_all)


def _sum_dim():
    x = torch.randn(64, 1000) * 0.5
    # 参照取 float64：CPU 的 sum/mean 用 pairwise 求和，比顺序累加准得多，
    # 拿 1e-5 当门槛会误判——顺序 fp32 累加 1000 项的固有相对误差本来就在 1e-5 量级。
    return x.to("corex").sum(dim=-1), x.double().sum(dim=-1)

check("sum(dim=-1) 64x1000（vs float64 参照）", _sum_dim, tol=1e-4)


def _mean_dim():
    x = torch.randn(64, 1000) * 0.5
    return x.to("corex").mean(dim=-1), x.double().mean(dim=-1)

check("mean(dim=-1) 64x1000（vs float64 参照）", _mean_dim, tol=1e-4)

print("=== 3. 索引（embedding 取行）===")

def _index_select():
    src = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32) * 0.25
    idx = torch.randint(0, 64, (7,), dtype=torch.int64)
    return torch.index_select(src.to("corex"), 0, idx.to("corex")), torch.index_select(src, 0, idx)

check("index_select 7 行（i64 下标）", _index_select, byte_equal=True)

print("=== 4. 拼一条注意力小块（形状按真实推理: T=16, H=64）===")

def _block():
    T, H = 16, 64
    x = torch.randn(T, H) * 0.3
    wq = torch.randn(H, H) * 0.1
    wk = torch.randn(H, H) * 0.1
    wv = torch.randn(H, H) * 0.1

    def fwd(xx, q, k, v, dev):
        q_ = xx @ q
        k_ = xx @ k
        v_ = xx @ v
        # k 转置：形状 [T,H] -> [H,T]，转置视图非连续，靠 .contiguous() 走一次拷贝
        scores = (q_ @ k_.t().contiguous()) / (H ** 0.5)
        p = torch.softmax(scores, dim=-1)
        return p @ v_

    ref = fwd(x, wq, wk, wv, "cpu")
    got = fwd(x.to("corex"), wq.to("corex"), wk.to("corex"), wv.to("corex"), "corex")
    return got, ref

check("注意力小块（含转置+contiguous）", _block, tol=1e-5, metric="global")

print("=== 5. 性能口径（含 Python 派发开销）===")
try:
    # 卡上没有 RNG（aten::normal_ 未实现），随机数先在 CPU 生成再搬上去。
    a = torch.randn(256, 256).to("corex")
    b = torch.randn(256, 256).to("corex")
    for _ in range(3):
        c = a @ b
    torch.corex.synchronize()
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        c = a @ b
    torch.corex.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"  mm 256x256x256 端到端 {dt * 1e6:.1f} us -> {2 * 256 ** 3 / dt / 1e9:.1f} GFLOP/s")
except Exception as exc:                                # noqa: BLE001
    print(f"  mm 计时失败：{type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

print("=== 6. 汇总 ===")
print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
for line in FAILED:
    print("  -", line)
print("TORCH_OPS_VERDICT", "PASS" if not FAILED else "PARTIAL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/ops_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
