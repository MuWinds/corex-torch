set -u
H=$HOME
V=$H/venvs/fork28
PY=$V/bin/python
TCL=$H/coffin-work/base/sandbox/work/fork/toolchain/lib
FORK=$H/work/fork/triton-corex
echo "HOST=$(hostname) DATE=$(date -Is)"
export PYTHONPATH=$FORK/python
export LD_PRELOAD=$TCL/libstdc++.so.6
export LD_LIBRARY_PATH=$TCL:${LD_LIBRARY_PATH:-}
"$PY" -c "
import torch
for name in ('cuda', 'corex', 'cuda:0', 'corex:0'):
    try:
        d = torch.device(name)
        print('DEVICE_OK', name, '->', d.type, d.index)
    except Exception as exc:
        print('DEVICE_FAIL', name, type(exc).__name__, str(exc)[:60])
x = torch.arange(4.0)
try:
    print('TO_CUDA', x.to('cuda').device)
except Exception as exc:
    print('TO_CUDA_FAIL', type(exc).__name__, str(exc)[:60])
try:
    print('TO_COREX', x.to('corex').device)
except Exception as exc:
    print('TO_COREX_FAIL', type(exc).__name__, str(exc)[:60])
" 2>&1 | grep -E "^DEVICE|^TO_" | head -8
echo "T207_DONE"
