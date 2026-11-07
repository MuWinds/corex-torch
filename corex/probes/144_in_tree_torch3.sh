set -u
# 目的：验证「torch 树内形态」能跑通——不重编 torch，先把 torch/corex 覆盖到已装的
# torch 2.8 CPU wheel 里（同版本），看 `import torch` 之后 `torch.device("corex")` 是否
# 直接可用。这是 fork 落地前的过渡验证：代码位置和 import 路径与 fork 里完全一致，
# 唯一的差别是「谁来 import 这个包」。
#
# 为什么要这么验：真正的 fork 要编 30–60 分钟才有结论；而这次改动动的是包路径与注册入口，
# 先用覆盖法把逻辑错掉的可能排掉，再花时间在编译上。
H=$HOME
PY=$H/venvs/corex28/bin/python
TGZ=$H/work/tcorex.tgz
echo "HOST=$(hostname) DATE=$(date -Is)"

SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "SITE=$SITE"
"$PY" -c "import torch; print('torch', torch.__version__)" || exit 66

echo "=== 1. 覆盖 torch/corex ==="
rm -rf "$SITE/corex"
mkdir -p "$SITE/corex"
tar xzf "$TGZ" -C "$SITE/corex" --strip-components=2 || exit 1
ls "$SITE/corex" | head -6

echo "=== 2. 把 shim 放回去（本次没打进包，避免把二进制带进源码树）==="
SO=$(find "$H/coffin-work" -name "_corex_shim.so" 2>/dev/null | head -1)
echo "找到 shim: ${SO:-无}"
if [ -n "$SO" ]; then cp -f "$SO" "$SITE/corex/_corex_shim.so"; fi

echo "=== 3. 给已装 torch 的 __init__.py 接入（幂等）==="
if grep -q "from . import corex as _corex" "$SITE/__init__.py"; then
    echo "已经接过了"
else
    cat >> "$SITE/__init__.py" <<'PYEOF'

# corex（天数智芯 天垓100）后端随 fork 一起发布：导入 torch 即完成 PrivateUse1 注册。
try:
    from . import corex as _corex  # noqa: F401

    _corex.register()
except Exception:
    pass
PYEOF
    echo "已接入"
fi

echo "=== 4. 功能验证 ==="
cat > "$H/work/tcorex_test.py" <<'PYEOF'
import torch

print("VERSION", torch.__version__)
print("HAS_COREX", hasattr(torch, "corex"))
print("AVAILABLE", torch.corex.is_available())
print("COUNT", torch.corex.device_count())

x = torch.zeros(4, 8, device="corex")
print("ZEROS", x.device, x.dtype, float(x.sum()))
a = torch.arange(8.0)
b = (a.to("corex") + 1.0)
print("ADD_OK", torch.equal(b.to("cpu"), a + 1.0))
print("STR", str(x.device))
PYEOF
cd "$H/work" && TRITON_COREX_TCU=0 "$PY" -u tcorex_test.py 2>&1 | head -30
echo "T142_DONE"
