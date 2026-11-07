set -u
# 目的：验证「客户照旧写 torch.cuda」这条路——这是本轮的核心交付。
#
# 判据（全部要过）：
#   torch.cuda.is_available() / device_count() / synchronize() 可用；
#   torch.version.cuda 不是 None、torch.backends.cuda.is_built() 为真；
#   `torch.zeros(4, device="cuda")`、`x.cuda()`、`x.is_cuda`、`x.device == cuda:0`；
#   `torch.device("corex")` 这种老写法也要还能用（兼容别名）；
#   算子在 cuda 写法下真的算对。
H=$HOME
W=$(ls $H/work/torchfork2/pytorch/dist/torch-*-cp310-*.whl 2>/dev/null | head -1)
V=$H/venvs/fork28
PY=$V/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -n "$W" ] || { echo "没有 cp310 wheel"; exit 66; }
echo "WHEEL=$W ($(stat -c %y "$W" | cut -d. -f1))"

echo "=== 1. 重装 wheel ==="
"$PY" -m pip install -q --no-deps --force-reinstall "$W" 2>&1 | tail -2
"$PY" -c "import torch; print('TORCH', torch.__version__)"

echo "=== 2. 重编 shim（针对这个 torch）==="
export PATH=$H/venvs/torchbuild310/bin:$PATH
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
"$PY" "$H/work/shim/build_shim.py" --src "$H/work/shim/corex_shim.cpp" \
      --out "$SITE/corex/_corex_shim.so" 2>&1 | tail -3

export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

echo "=== 3. torch.cuda 这一面 ==="
"$PY" -c "
import torch
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_COUNT', torch.cuda.device_count())
print('VERSION_CUDA', torch.version.cuda)
print('IS_BUILT', torch.backends.cuda.is_built())
x = torch.arange(8.0)
d = x.cuda()
print('X_CUDA', d.device, 'IS_CUDA', d.is_cuda, 'DTYPE', d.dtype)
torch.cuda.synchronize()
z = torch.zeros(4, 8, device='cuda')
print('ZEROS_CUDA', z.device, float(z.sum()))
y = (x.cuda() + 1.0)
print('ADD_CUDA', bool(torch.equal(y.cpu(), x + 1.0)))
print('CURRENT_DEVICE', torch.cuda.current_device())
s = torch.cuda.Stream()
print('STREAM', type(s).__name__)
e = torch.cuda.Event()
print('EVENT', type(e).__name__)
print('DEVICE_STR_COREX', torch.device('corex').type)
" 2>&1 | tail -14

echo "=== 4. 算子矩阵（沿老写法 device='corex'）+ mini decoder ==="
( cd "$H/work" && "$PY" -u opsmatrix2.py 2>&1 | grep -E "^(FAIL|MISMATCH|SUMMARY)" | tail -5 )
( cd "$H/work" && "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax" | head -3 )
echo "T204_DONE"
