set -u
# 目的：最终确认「装完即用」的 wheel 在真机上可用。
#
# 为什么把结果写到 ~/work/final_check.txt：沙箱队列的 logs/<name>.out 在多个人共用的环境下
# 出现过读到别人/旧内容的情况，用唯一文件最稳。
H=$HOME
OUT=$H/work/final_check.txt
PY=$H/venvs/fresh28/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

: > "$OUT"
{
  echo "HOST=$(hostname) DATE=$(date -Is)"
  SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null) || exit 66
  echo "SHIM_FROM_WHEEL=$(stat -c %s "$SITE/corex/_corex_shim.so" 2>/dev/null || echo 缺)"
  echo "SYMBOL=$(nm -D --defined-only "$SITE/corex/_corex_shim.so" 2>/dev/null | grep -o 'PyInit_[A-Za-z_0-9]*' | head -1)"

  echo "--- 设备 ---"
  timeout 90 "$PY" -c "
import torch
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_COUNT', torch.cuda.device_count())
x = torch.arange(8.0)
d = x.cuda()
print('X_CUDA', d.device, 'IS_CUDA', d.is_cuda)
print('ADD_CUDA', bool(torch.equal((d + 1.0).cpu(), x + 1.0)))
torch.cuda.synchronize()
print('STREAM', type(torch.cuda.Stream()).__name__)
print('VERSION_CUDA', torch.version.cuda, 'IS_BUILT', torch.backends.cuda.is_built())
" 2>&1 | tail -8
  echo "DEVICE_RC=$?"

  echo "--- 算子矩阵 ---"
  ( cd "$H/work" && timeout 600 "$PY" -u opsmatrix2.py 2>&1 | grep -E "^(FAIL|MISMATCH|SUMMARY)" | tail -4 )
  echo "--- mini decoder ---"
  ( cd "$H/work" && timeout 600 "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax" | head -3 )
  echo "FINAL_CHECK_DONE"
} 2>&1 | tee -a "$OUT"
