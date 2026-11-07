set -u
# 目的：验收第 6 批（中间的归约与方差）：`sum(dim=[中间维])`、`mean(dim)`、`var`、`std`。
# 这批是 probes/120 里剩下的内核缺口，内核侧已由 probes/197 验过（PASS），这里验「分派接得对」：
#   - 归约维在**中间**（不是最后一维、也不是全归约），形状要覆盖 inner>1；
#   - `correction` 的两种取值（1 = 样本、0 = 总体）；
#   - 与 CPU 参照比，判据按「能逐字节就逐字节」。
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

cat > "$W/axis_reduce.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch

corex_torch.register()
torch.manual_seed(11)
FAILED, PASSED = [], []


def dev(t):
    return t.to("corex")


def check(name, fn, byte_equal=True, tol=1e-5):
    try:
        got, ref = fn()
        if tuple(got.shape) != tuple(ref.shape):
            raise AssertionError(f"形状不一致 {tuple(got.shape)} vs {tuple(ref.shape)}")
        if byte_equal:
            ok = torch.equal(got.cpu(), ref)
            extra = "byte-equal" if ok else "字节不等"
        else:
            d = float((got.cpu().double() - ref.double()).abs().max())
            scale = float(ref.double().abs().max()) or 1.0
            ok = d / scale <= tol
            extra = f"max_abs={d:.3e} 全局相对={d / scale:.3e}（门槛 {tol:.0e}）"
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:44s} {extra}")
        (PASSED if ok else FAILED).append(name)
    except Exception as exc:                            # noqa: BLE001
        print(f"  [FAIL] {name:44s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        FAILED.append(f"{name} -> {type(exc).__name__}")


print("=== 1. sum(dim=中间维)：inner > 1 才检验得出跨步读对不对 ===")
for shape, dim in (((4, 8, 16), 1), ((4, 8, 16), 0), ((2, 3, 4, 5), 2), ((4, 1000), 0)):
    x = torch.randn(*shape)

    def fn(x=x, dim=dim):
        return dev(x).sum(dim=dim), x.sum(dim=dim)

    check(f"sum dim={dim} shape={shape}（与 CPU 比）", fn, byte_equal=False, tol=1e-6)


def _my_dyadic(shape, dim):
    """dyadic 数据上 sum 应当逐字节相等（fp32 加法在 2 的幂上是精确的）。"""
    x = torch.randint(-4, 5, shape).float()
    return dev(x).sum(dim=dim), x.sum(dim=dim)


check("sum dim=1 4x8x16（dyadic，应逐字节）", lambda: _my_dyadic((4, 8, 16), 1), byte_equal=True)

print("=== 2. sum(dim) 带 keepdim ===")


def _keepdim():
    x = torch.randint(-4, 5, (4, 8, 16)).float()
    return dev(x).sum(dim=1, keepdim=True), x.sum(dim=1, keepdim=True)


check("sum dim=1 keepdim=True 4x8x16", _keepdim, byte_equal=True)

print("=== 3. mean(dim=中间维) / 最后一维 / 全归约 ===")
for shape, dim in (((4, 8, 16), 1), ((4, 8, 16), 2)):
    x = torch.randn(*shape)

    def fn(x=x, dim=dim):
        return dev(x).mean(dim=dim), x.mean(dim=dim)

    check(f"mean dim={dim} shape={shape}", fn, byte_equal=False, tol=1e-6)


def _mean_all():
    x = torch.randn(4, 8, 16)
    return dev(x).mean(), x.mean()


check("mean() 全归约 4x8x16", _mean_all, byte_equal=False, tol=1e-6)

print("=== 4. var / std（含 correction=0 与 1）===")
for corr in (1, 0):
    x = torch.randn(32, 1000) * 2 + 5

    def fn(x=x, corr=corr):
        return dev(x).var(dim=-1, correction=corr), x.var(dim=-1, correction=corr)

    check(f"var(dim=-1, correction={corr}) 32x1000", fn, byte_equal=False, tol=1e-5)

    def fn_std(x=x, corr=corr):
        return dev(x).std(dim=-1, correction=corr), x.std(dim=-1, correction=corr)

    check(f"std(dim=-1, correction={corr}) 32x1000", fn_std, byte_equal=False, tol=1e-5)

print("=== 5. 汇总 ===")
print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
for line in FAILED:
    print("  -", line)
print("AXIS_REDUCE_VERDICT", "PASS" if not FAILED else "PARTIAL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/axis_reduce.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
