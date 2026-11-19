#!/bin/bash
V=$HOME/venvs/torch24test
cd $HOME
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