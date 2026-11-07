set -u
# 目的：验证「装完即用」——wheel 自带 torch/corex/_corex_shim.so，装完不跑任何编译脚本就能用卡。
#
# 判据：
#   1. 用**全新 venv**（没有历史 shim）；
#   2. 只做 pip install，不调用 build_shim.py；
#   3. torch.cuda 那一面全部可用，算子矩阵与 mini decoder 全过。
H=$HOME
W=$(ls -t $H/work/torchfork2/pytorch/dist/torch-*-cp310-*.whl 2>/dev/null | head -1)
V=$H/venvs/fresh28
PY=$V/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -n "$W" ] || { echo "没有 cp310 wheel"; exit 66; }
echo "WHEEL=$(basename "$W")  $(stat -c %s "$W") 字节"

echo "=== 1. 全新 venv，只装 wheel（不编任何东西）==="
rm -rf "$V"
"$H/venvs/torchbuild310/bin/python" -m venv "$V" || exit 66
"$PY" -m pip install -q -U pip 2>&1 | tail -1
"$PY" -m pip install -q numpy typing_extensions filelock jinja2 networkx sympy requests six 2>&1 | tail -2
"$PY" -m pip install -q --no-deps "$W" 2>&1 | tail -2
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "shim 在包里: $(ls -la "$SITE/corex/_corex_shim.so" | awk '{print $5}') 字节"

echo "=== 2. 直接 import 验证（没有任何编译步骤）==="
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
"$PY" -c "
import torch
print('TORCH', torch.__version__)
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_COUNT', torch.cuda.device_count())
x = torch.arange(8.0)
d = x.cuda()
print('X_CUDA', d.device, 'IS_CUDA', d.is_cuda)
print('ADD_CUDA', bool(torch.equal((d + 1.0).cpu(), x + 1.0)))
torch.cuda.synchronize()
print('STREAM', type(torch.cuda.Stream()).__name__)
" 2>&1 | tail -8

echo "=== 3. 算子矩阵 + mini decoder ==="
( cd "$H/work" && "$PY" -u opsmatrix2.py 2>&1 | grep -E "^(FAIL|MISMATCH|SUMMARY)" | tail -4 )
( cd "$H/work" && "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax" | head -3 )
echo "T218_DONE"
