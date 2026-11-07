set -u
# 目的：验证刚编出来的 cp312 wheel 是真 torch（能 import、能算、带 torch.corex）。
# 这一步只验「构建产物是有效的」，与卡片无关；py310 的版本另起一轮编。
H=$HOME
W=$(ls $H/work/torchfork2/pytorch/dist/torch-*.whl | head -1)
V=$H/venvs/whlcheck
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "WHEEL=$W"
python3 -m venv "$V" 2>/dev/null || true
"$V/bin/pip" install -q -U pip 2>&1 | tail -1
"$V/bin/pip" install -q numpy typing_extensions filelock jinja2 networkx sympy 2>&1 | tail -2
"$V/bin/pip" install -q --no-deps "$W" 2>&1 | tail -3
echo "=== import 验证 ==="
"$V/bin/python" -c "
import torch
print('VERSION', torch.__version__)
print('HAS_COREX_SUBMODULE', 'corex' in dir(torch))
x = torch.randn(64, 64)
print('CPU_MM_OK', bool((x.mm(x).isfinite().all())))
print('BACKENDS', torch.backends.mkldnn.is_available())
" 2>&1 | tail -8
echo "T194_DONE"
