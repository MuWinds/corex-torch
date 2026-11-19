#!/bin/bash
V=$HOME/venvs/torch24test
cd $HOME
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex-3.2.1/lib:$LD_LIBRARY_PATH
$V/bin/python - <<'EOF'
import torch
print("version", torch.__version__)
print("cuda", torch.version.cuda, "corex", getattr(torch, "corex", None))
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("matmul", tuple(y.shape), float(y.sum()))
import torch.nn as nn
m = nn.Linear(256, 256).cuda()
o = m(torch.randn(8, 256, device="cuda"))
print("linear", tuple(o.shape))
EOF
echo py_rc=$?