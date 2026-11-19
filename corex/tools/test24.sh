#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
V=$HOME/venvs/torch24test
rm -rf $V
python3.10 -m venv --system-site-packages $V
$V/bin/pip install --no-deps --force-reinstall $HOME/build/torch24/dist/torch-2.4.0+corex-cp310-cp310-linux_x86_64.whl > $HOME/build/test24_install.log 2>&1
echo install_rc=$?
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex-3.2.1/lib:$LD_LIBRARY_PATH
$V/bin/python - <<'EOF'
import torch, torch.corex
print("version", torch.__version__)
print("corex", torch.corex.corex, torch.corex.corex_arch)
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("matmul", tuple(y.shape), float(y.sum()))
EOF
echo py_rc=$?