set -u
# 目的：让 torch 这条线用上 **triton fork**（而不是插件 triton_corex），验证 elementwise 热路径
# 真的能编出来并跑对。
#
# 关键环境：
#   * PYTHONPATH 指到 fork 的 python 目录 → `import triton` 拿到 fork（含 triton.backends.corex）；
#   * LD_PRELOAD 新 libstdc++ → fork 的 libtriton.so 和 mlir-tblgen 都要它；
#   * COREX_TRITON_SRC 不设（fork 形态不需要插件源码目录）。
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
echo "SITE=$SITE shim=${SO:-无}"

export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

echo "=== 1. fork 后端是否可见 ==="
"$PY" -c "
import triton, triton.backends
print('triton', triton.__version__, triton.__file__)
print('backends', sorted(triton.backends.backends))
import triton.backends.corex.compiler as c
print('corex compiler at', c.__file__)
" 2>&1 | tail -5

echo "=== 2. torch 侧功能验证 ==="
cat > "$H/work/t4.py" <<'PYEOF'
import traceback
import torch

print("REG", torch.corex.register() is None, torch.corex.is_available(), torch.corex.device_count(), flush=True)
x = torch.zeros(4, 8, device="corex")
print("ZEROS", x.device, x.dtype, float(x.sum()), flush=True)
a = torch.arange(8.0)
b = a.to("corex") + 1.0
print("ADD", torch.equal(b.to("cpu"), a + 1.0), flush=True)
c = (a.to("corex") * 2.0).to("cpu")
print("MUL", torch.equal(c, a * 2.0), flush=True)
PYEOF
cd "$H/work" && "$PY" -u t4.py > "$H/work/t4.out" 2>&1
grep -nE "^REG|^ZEROS|^ADD|^MUL|Error:|RuntimeError:" "$H/work/t4.out" | head -10 | cut -c1-200
echo "--- 尾部 ---"
tail -4 "$H/work/t4.out" | cut -c1-200
echo "T148_DONE"
