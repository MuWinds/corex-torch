set -u
# 目的：在远端重编 C++ shim（刚修了 materialize_broadcast 里 pitch=0 的广播物化），
# 然后重跑功能矩阵，确认 layer_norm 带 bias 的路径通了。
#
# 远端没有仓库树，所以 build_shim.py 用 --src/--out 显式指路径（它的默认值是按仓库布局推的）。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
SHIMDIR=$H/work/shim
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66

# cpp_extension 找 ninja 是 `which ninja`，所以要把 venv 的 bin 放进 PATH
export PATH=$H/venvs/corex28/bin:$PATH
echo "=== 1. 编译 shim ==="
"$PY" "$SHIMDIR/build_shim.py" --src "$SHIMDIR/corex_shim.cpp" \
      --out "$SITE/corex/_corex_shim.so" 2>&1 | tail -6
ls -la "$SITE/corex/_corex_shim.so" 2>/dev/null || { echo "没编出 .so"; exit 67; }

echo "=== 2. 覆盖包（含本轮的 bf16 与 memcpy 防御）==="
rm -rf "$SITE/corex_tmp"; mkdir -p "$SITE/corex_tmp"
tar xzf "$H/work/tcorex5.tgz" -C "$SITE/corex_tmp" --strip-components=2 || exit 1
cp -f "$SITE/corex_tmp"/*.py "$SITE/corex/" 2>/dev/null
rm -rf "$SITE/corex_tmp"

echo "=== 3. 功能矩阵 ==="
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
( cd "$H/work" && "$PY" -u matrix.py 2>&1 | grep -E "^(OK|MISMATCH|FAIL|SUMMARY)" | head -30 )
echo "T154_DONE"
