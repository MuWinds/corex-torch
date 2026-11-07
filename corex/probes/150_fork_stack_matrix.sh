set -u
# 目的：在「torch(树内 corex) + triton fork」这套新栈上，一次拿到两样东西：
#   1. bf16 快路径编译失败的**完整**原因（之前只看到被截断的 `llc -marc...`）；
#   2. 一张功能矩阵：哪些算子已经通、哪些还没。
#
# 为什么自己写矩阵而不直接跑旧探针：旧探针都是按插件形态写的（`sys.path` 加 `src`、`import
# corex_torch`），现在后端在 torch 树里，导入方式变了；矩阵脚本只依赖 `import torch`，
# 换形态不用改。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
rm -rf "$SITE/corex"; mkdir -p "$SITE/corex"
tar xzf "$H/work/tcorex4.tgz" -C "$SITE/corex" --strip-components=2 || exit 1
SO=$(find "$H/coffin-work" -name "_corex_shim.so" 2>/dev/null | head -1)
[ -n "$SO" ] && cp -f "$SO" "$SITE/corex/_corex_shim.so"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

echo "=== 1. bf16 编译失败的完整报错 ==="
cat > "$H/work/bf16_diag.py" <<'PYEOF'
import torch
import torch.corex  # noqa: F401

try:
    a = torch.ones(16, dtype=torch.bfloat16, device="corex")
    b = a + 1.0
    print("BF16_ADD_OK", b.to("cpu")[0].item())
except Exception as exc:
    print("BF16_ADD_FAIL", type(exc).__name__)
    print(str(exc)[:3000])
PYEOF
( cd "$H/work" && "$PY" -u bf16_diag.py 2>&1 | grep -vE "^\s*$" | tail -25 | cut -c1-220 )

echo
echo "=== 2. 功能矩阵 ==="
cat > "$H/work/matrix.py" <<'PYEOF'
import torch

torch.corex.register()

results = []


def check(name, fn):
    try:
        ok = fn()
        results.append((name, "OK" if ok else "MISMATCH"))
    except Exception as exc:
        msg = str(exc).split("\n")[0][:90]
        results.append((name, f"FAIL {type(exc).__name__}: {msg}"))


def dev(t):
    return t.to("corex")


check("zeros", lambda: float(torch.zeros(4, 4, device="corex").sum()) == 0.0)
check("empty+fill", lambda: float(torch.ones(8, device="corex").sum()) == 8.0)
check("arange", lambda: torch.equal(dev(torch.arange(6.0)).to("cpu"), torch.arange(6.0)))
check("h2d/d2h", lambda: torch.equal(dev(torch.arange(6.0)).to("cpu"), torch.arange(6.0)))


def add_t():
    a = torch.randn(64, device="cpu")
    b = torch.randn(64, device="cpu")
    return torch.allclose((dev(a) + dev(b)).to("cpu"), a + b, atol=1e-5)


def add_s():
    a = torch.randn(64)
    return torch.allclose((dev(a) + 2.0).to("cpu"), a + 2.0, atol=1e-5)


check("add.tensor", add_t)
check("add.scalar", add_s)
check("sub", lambda: torch.allclose((dev(torch.arange(8.0)) - 1).to("cpu"), torch.arange(8.0) - 1))
check("mul", lambda: torch.allclose((dev(torch.arange(8.0)) * 3).to("cpu"), torch.arange(8.0) * 3))
check("div", lambda: torch.allclose((dev(torch.arange(1.0, 9.0)) / 2).to("cpu"), torch.arange(1.0, 9.0) / 2))


def mm():
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    return torch.allclose(dev(a).mm(dev(b)).to("cpu"), a.mm(b), atol=1e-3)


def bmm():
    a = torch.randn(2, 16, 16)
    b = torch.randn(2, 16, 16)
    return torch.allclose(torch.bmm(dev(a), dev(b)).to("cpu"), torch.bmm(a, b), atol=1e-3)


check("mm", mm)
check("bmm", bmm)


def softmax():
    a = torch.randn(8, 32)
    return torch.allclose(dev(a).softmax(-1).to("cpu"), a.softmax(-1), atol=1e-5)


def sum_():
    a = torch.randn(8, 32)
    return torch.allclose(dev(a).sum().to("cpu"), a.sum(), atol=1e-4)


def sum_dim():
    a = torch.randn(8, 32)
    return torch.allclose(dev(a).sum(-1).to("cpu"), a.sum(-1), atol=1e-4)


def mean_():
    a = torch.randn(8, 32)
    return torch.allclose(dev(a).mean().to("cpu"), a.mean(), atol=1e-5)


check("softmax", softmax)
check("sum", sum_)
check("sum.dim", sum_dim)
check("mean", mean_)


def layer_norm():
    a = torch.randn(4, 32)
    w = torch.randn(32)
    b = torch.randn(32)
    ref = torch.nn.functional.layer_norm(a, (32,), w, b)
    got = torch.nn.functional.layer_norm(dev(a), (32,), dev(w), dev(b))
    return torch.allclose(got.to("cpu"), ref, atol=1e-4)


def view_t():
    a = torch.randn(4, 8)
    return torch.allclose(dev(a).view(32).to("cpu"), a.view(32))


def transpose_t():
    a = torch.randn(4, 8)
    return torch.allclose(dev(a).transpose(0, 1).to("cpu"), a.transpose(0, 1))


def relu_t():
    a = torch.randn(64)
    return torch.allclose(torch.relu(dev(a)).to("cpu"), torch.relu(a))


def cat_t():
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    return torch.allclose(torch.cat([dev(a), dev(b)], 0).to("cpu"), torch.cat([a, b], 0))


check("layer_norm", layer_norm)
check("view", view_t)
check("transpose", transpose_t)
check("relu", relu_t)
check("cat", cat_t)


def matmul_2d():
    a = torch.randn(16, 32)
    b = torch.randn(32, 24)
    return torch.allclose((dev(a) @ dev(b)).to("cpu"), a @ b, atol=1e-3)


check("matmul", matmul_2d)


def bf16_roundtrip():
    a = torch.randn(16).to(torch.bfloat16)
    return torch.equal(dev(a).to("cpu"), a)


check("bf16.copy", bf16_roundtrip)

ok = sum(1 for _, s in results if s == "OK")
for name, status in results:
    print(f"{status:<40} {name}")
print(f"SUMMARY {ok}/{len(results)} OK")
PYEOF
( cd "$H/work" && "$PY" -u matrix.py 2>&1 | grep -E "^(OK|MISMATCH|FAIL|SUMMARY)" | head -30 )
echo "T150_DONE"
