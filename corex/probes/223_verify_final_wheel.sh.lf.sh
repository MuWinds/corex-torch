set -u
# 目的：验证「装完即用」的最终 wheel（自带上修好的 shim）在真机上是否可用。
#
# 卡的健康状态会飘（reprobe 在 healthy=6,7 与 healthy=none 之间来回），而**初始化死卡会挂住**，
# 一旦挂住就占满串行队列 1800 s。所以每个碰卡的命令都包一层 `timeout`：不健康就快速退出。
H=$HOME
PY=$H/venvs/fresh28/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
SITE=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__))") || exit 66
echo "shim 来自 wheel=$(stat -c %s "$SITE/corex/_corex_shim.so") 字节"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}

echo "=== 卡可用性（最多等 90 s）==="
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
" 2>&1 | tail -8
RC=$?
echo "上一步 rc=$RC（124 = 超时，通常是卡不健康）"
if [ "$RC" -ne 0 ]; then echo "SKIP_REST 卡不可用"; echo "T223_DONE"; exit 0; fi

echo "=== 算子矩阵 + mini decoder ==="
( cd "$H/work" && timeout 600 "$PY" -u opsmatrix2.py 2>&1 | grep -E "^(FAIL|MISMATCH|SUMMARY)" | tail -4 )
( cd "$H/work" && timeout 600 "$PY" -u mini_decoder.py 2>&1 | grep -E "VERDICT|相对误差|argmax" | head -3 )
echo "T223_DONE"
