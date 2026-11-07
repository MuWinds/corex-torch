set -u
# 目的：把 fork 编出来的 **py310 wheel** 装进一个干净 venv，然后端到端验一遍。
#
# 这一步是「fork 真的成了」的判据：
#   * wheel 里带 `torch/corex`（构建时已经从源码树里打进去）；
#   * `import torch` 之后 `torch.device("corex")` 直接可用（不需要任何外挂包）；
#   * C++ shim 针对**这个** torch 重编（pybind11 模块与 torch 的 ABI 必须对齐）；
#   * 功能矩阵、算子矩阵、mini decoder 全过。
H=$HOME
W=$(ls $H/work/torchfork2/pytorch/dist/torch-*-cp310-*.whl 2>/dev/null | head -1)
V=$H/venvs/fork28
PY=$V/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -n "$W" ] || { echo "还没有 cp310 的 wheel"; exit 66; }
echo "WHEEL=$W"

echo "=== 1. 干净 venv + 装 wheel ==="
"$PY" --version 2>/dev/null || "$H/venvs/torchbuild310/bin/python" -m venv "$V"
"$PY" -m pip install -q -U pip wheel 2>&1 | tail -1
"$PY" -m pip install -q numpy typing_extensions filelock jinja2 networkx sympy requests six 2>&1 | tail -2
"$PY" -m pip install -q --no-deps --force-reinstall "$W" 2>&1 | tail -3
"$PY" -c "import torch; print('TORCH', torch.__version__, torch.__file__)"

echo "=== 2. 针对这个 torch 重编 C++ shim ==="
export PATH=$H/venvs/torchbuild310/bin:$PATH
"$PY" -m pip install -q ninja 2>&1 | tail -1
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
"$PY" "$H/work/shim/build_shim.py" --src "$H/work/shim/corex_shim.cpp" \
      --out "$SITE/corex/_corex_shim.so" 2>&1 | tail -4

echo "=== 3. 设备就绪检查 ==="
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
"$PY" -c "
import torch
print('AVAILABLE', torch.corex.is_available(), 'COUNT', torch.corex.device_count())
x = torch.zeros(4, 8, device='corex')
a = torch.arange(8.0)
b = a.to('corex') + 1.0
print('ZEROS', x.device, float(x.sum()))
print('ADD', bool(torch.equal(b.to('cpu'), a + 1.0)))
" 2>&1 | tail -6

echo "=== 4. 算子矩阵 + mini decoder ==="
( cd "$H/work" && "$PY" -u opsmatrix2.py 2>&1 | grep -E "^(FAIL|MISMATCH|SUMMARY)" | tail -6 )
( cd "$H/work" && "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax" | head -4 )
echo "T197_DONE"
