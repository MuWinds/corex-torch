set -u
# 目的：把「算子清单」升级成「一条真实前向」——在**新栈**（torch 树内 corex + triton fork）上
# 跑 mini decoder，和 CPU 参照比全局相对误差。
#
# payload 直接从 probes/018 抽出来的同一份代码，只去掉「插件时代」的三行
# （sys.path.insert / import corex_torch / corex_torch.register()）——现在 import torch
# 就自动注册了。这样两套形态跑的是同一条前向，数字可以直接对。
H=$HOME
PY=$H/venvs/corex28/bin/python
FORK=$H/work/fork/triton-corex
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
rm -rf "$SITE/corex/doc_tmp" 2>/dev/null
# 用当前仓库里的包（上一轮已经覆盖过，这里再覆盖一次保证是最新的）
rm -rf "$SITE/corex"; mkdir -p "$SITE/corex"
tar xzf "$H/work/tcorex5.tgz" -C "$SITE/corex" --strip-components=2 || exit 1
cp -f "$H/work/shim_built.so" "$SITE/corex/_corex_shim.so" 2>/dev/null || \
  cp -f "$(find "$H/coffin-work" -name "_corex_shim.so" 2>/dev/null | head -1)" "$SITE/corex/_corex_shim.so"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
echo "=== mini decoder（新栈）==="
cd "$H/work" && "$PY" -u mini_decoder.py 2>&1 | grep -vE "UserWarning|_warn_fast_op_once" | tail -14
echo "T157_DONE"
