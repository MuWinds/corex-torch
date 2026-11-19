#!/bin/bash
INV=$HOME/env_inventory_20261119.txt
> $INV
for v in $HOME/venvs/*/; do
  echo "=== $(basename $v)" >> $INV
  $v/bin/pip freeze >> $INV 2>/dev/null
done
echo "=== conda_py312" >> $INV
$HOME/.conda/envs/py312/bin/pip freeze >> $INV 2>/dev/null
echo "inventory_lines=$(wc -l < $INV)"
rm -rf $HOME/venvs/*
rm -rf $HOME/.conda/envs/py312
echo "after_rm venvs=$(ls -1 $HOME/venvs 2>/dev/null | wc -l) py312=$(test -d $HOME/.conda/envs/py312 && echo exists || echo gone)"
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python3.10
mkdir -p $HOME/envs
rm -rf $HOME/envs/torch24
$PY -m venv --system-site-packages $HOME/envs/torch24
echo "venv_rc=$?"
$HOME/envs/torch24/bin/pip install --no-deps --force-reinstall $HOME/build/dist24/torch-2.4.0+corex-cp310-cp310-linux_x86_64.whl > $HOME/env_install.log 2>&1
echo "install_rc=$?"
cd $HOME
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex-3.2.1/lib:$LD_LIBRARY_PATH
$HOME/envs/torch24/bin/python - <<'EOF'
import torch
print("version", torch.__version__, "cuda", torch.version.cuda)
print("avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
x = torch.randn(512, 512, device="cuda")
print("matmul", float((x @ x).sum()))
EOF
echo "py_rc=$?"
du -sh $HOME/envs/torch24