set -u
# 目的：确认沙箱里 `import corex_torch` 到底加载的是哪一份源码。
# 症状：运行期报的错在 `$W/src` 里已经不存在了（连续两次），必须先排除「加载了别处的旧副本」。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
"$PY" -B -c "
import sys
sys.path.insert(0, '$W/src')
import corex_torch, inspect
from corex_torch import _device
print('corex_torch =', corex_torch.__file__)
print('_device     =', _device.__file__)
src = inspect.getsource(_device)
for probe in ('暂不支持非连续的目的张量', '线性 D2H 到连续暂存'):
    print(f'  源码含 {probe!r}:', probe in src)
print('sys.path[:6] =', sys.path[:6])
"
echo "--- 磁盘上还有几份 ---"
find $HOME/coffin-work/base/sandbox/work -name _device.py -path '*corex_torch*' 2>/dev/null | head -5
