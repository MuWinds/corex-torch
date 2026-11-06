set -u
# 目的：第 5 批算子（relu/rsqrt/gelu/masked_fill/log_softmax/max.dim/argmax/view.dtype）
# 在 **torch 层**的验收。内核层已由 probes/196 验过（63/63+36/36），这一层只验「分发接得对不对」：
# 形状/keepdim/原地语义/返回 namedtuple/下标的 dtype 与取值。
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

cat > "$W/ops5_check.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import torch
import torch.nn.functional as F

import corex_torch

corex_torch.register()
torch.manual_seed(0)

FAILED, PASSED = [], []


def report(name, ok, extra=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:44s} {extra}")
    (PASSED if ok else FAILED).append(name)


def maxdiff(a, b):
    return float((a.cpu().double() - b.cpu().double()).abs().max())


def run(name, fn, tol=0.0, byte_equal=False):
    """tol=0 且 byte_equal=True 时要求逐字节相等；否则用 max_abs 与 tol 比。"""
    try:
        got, ref = fn()
        if byte_equal:
            ok = torch.equal(got.cpu(), ref)
            report(name, ok, "byte-equal" if ok else "字节不等")
        else:
            d = maxdiff(got, ref)
            report(name, d <= tol, f"max_abs={d:.3e}（门槛 {tol:.0e}）")
    except Exception as exc:                            # noqa: BLE001
        report(name, False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}")


def dev(t):
    return t.to("corex")


print("=== 1. relu / relu_ / rsqrt ===")


def _relu():
    x = -torch.randn(32, 512)
    return torch.relu(dev(x)), torch.relu(x)


run("relu 32x512", _relu, byte_equal=True)


def _relu_inplace():
    x = -torch.randn(32, 512)
    xd = dev(x).clone()
    xd.relu_()
    return xd, torch.relu(x)


run("relu_ 原地 32x512", _relu_inplace, byte_equal=True)


def _rsqrt():
    x = torch.rand(32, 512) + 0.5
    return torch.rsqrt(dev(x)), torch.rsqrt(x)


run("rsqrt 32x512", _rsqrt, tol=1e-6)

print("=== 2. gelu（两种近似分别与同一条公式比）===")


def _gelu(approximate):
    # 关键：got / ref 必须用**同一份输入**，否则测到的是两次随机采样的差（第一版就错了）。
    x = torch.randn(32, 512)
    return F.gelu(dev(x), approximate=approximate), F.gelu(x, approximate=approximate)


run("gelu erf 32x512", lambda: _gelu("none"), tol=1e-6)
run("gelu tanh 32x512", lambda: _gelu("tanh"), tol=1e-6)

print("=== 3. masked_fill ===")


def _masked():
    x = torch.randn(32, 512)
    m = torch.rand(32, 512) < 0.4
    return dev(x).masked_fill(dev(m), -7.5), x.masked_fill(m, -7.5)


run("masked_fill 32x512", _masked, byte_equal=True)

print("=== 4. log_softmax ===")


def _log_softmax(shape):
    x = torch.randn(*shape) * 2
    return torch.log_softmax(dev(x), dim=-1), torch.log_softmax(x, dim=-1)


run("log_softmax 32x512", lambda: _log_softmax((32, 512)), tol=1e-5)
run("log_softmax 2x8x256", lambda: _log_softmax((2, 8, 256)), tol=1e-5)

print("=== 5. max.dim / argmax ===")


def _max_dim():
    x = torch.randn(32, 512)
    v, i = dev(x).max(dim=-1)
    rv, ri = x.max(dim=-1)
    ok_v = torch.equal(v.cpu(), rv)
    ok_i = torch.equal(i.cpu(), ri) and i.dtype == torch.int64
    report("max(dim=-1) 值与下标", ok_v and ok_i,
           f"values_equal={ok_v} indices_equal={ok_i} idx_dtype={i.dtype}")
    return v, rv


run("max(dim=-1) 值", _max_dim, byte_equal=True)


def _argmax():
    x = torch.randn(32, 1000)
    return torch.argmax(dev(x), dim=-1), torch.argmax(x, dim=-1)


run("argmax(dim=-1) 32x1000", _argmax, byte_equal=True)

print("=== 6. view.dtype ===")


def _view_dtype():
    x = torch.randn(64, device="cpu")
    v = dev(x).view(torch.int32)
    ref = x.view(torch.int32)
    print(f"        got: dtype={v.dtype} shape={tuple(v.shape)} "
          f"first3={v.cpu()[:3].tolist()}")
    print(f"        ref: dtype={ref.dtype} shape={tuple(ref.shape)} "
          f"first3={ref[:3].tolist()}")
    return v, ref


run("view.dtype fp32->i32（同宽重解释）", _view_dtype, byte_equal=True)

print("=== 7. 汇总 ===")
print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
for line in FAILED:
    print("  -", line)
print("OPS5_VERDICT", "PASS" if not FAILED else "PARTIAL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/ops5_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
