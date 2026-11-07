set -u
# 目的：把 layer_norm 里那条 `cudaMemcpy2D(D2D) invalid argument` 的**实参**抓出来。
# 之前的两个防御性判断（pitch>0 / 空块）没挡住，说明它来自另一条调用点，得看参数。
#
# 手法：把 shim 模块的 memcpy2d 包一层，先打印实参再调用；异常时把最后一条参数原样留下。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
rm -rf "$SITE/corex"; mkdir -p "$SITE/corex"
tar xzf "$H/work/tcorex5.tgz" -C "$SITE/corex" --strip-components=2 || exit 1
SO=$(find "$H/coffin-work" -name "_corex_shim.so" 2>/dev/null | head -1)
[ -n "$SO" ] && cp -f "$SO" "$SITE/corex/_corex_shim.so"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

cat > "$H/work/ln_diag.py" <<'PYEOF'
import torch
import torch.corex
from torch.corex import _device

torch.corex.register()
shim = _device.shim()
orig = shim.memcpy2d
calls = []


def wrapped(dst, dpitch, src, spitch, width, height, dev):
    calls.append((dst, dpitch, src, spitch, width, height, dev))
    return orig(dst, dpitch, src, spitch, width, height, dev)


shim.memcpy2d = wrapped

a = torch.randn(4, 32)
w = torch.randn(32)
b = torch.randn(32)
try:
    out = torch.nn.functional.layer_norm(a.to("corex"), (32,), w.to("corex"), b.to("corex"))
    print("LN_OK", float(out.sum().to("cpu")))
except Exception as exc:
    print("LN_FAIL", type(exc).__name__, str(exc)[:150])
print("NGPU_CALLS", len(calls))
for i, c in enumerate(calls[-6:]):
    print("CALL", i, c[1], c[3], c[4], c[5], c[6])
PYEOF
( cd "$H/work" && "$PY" -u ln_diag.py 2>&1 | grep -E "^LN_|^NGPU_CALLS|^CALL" | head -12 )
echo "T152_DONE"
