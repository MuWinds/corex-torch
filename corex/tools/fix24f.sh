#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
rm -f caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/TensorTopK.*.o
ninja -k 20 > $HOME/build/fix24f.log 2>&1
echo ninja_rc=$? failed=$(grep -c FAILED: $HOME/build/fix24f.log)
cd $HOME/build/torch24 || exit 0
printf '2.4.0+corex\n' > version.txt
printf "corex = True\ncorex_arch = 'ivcore10'\n" > torch/corex.py
export USE_CUDA=1 USE_CUDNN=0 USE_KINETO=0 USE_NCCL=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_NNPACK=0 USE_XNNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
export USE_FBGEMM=0 USE_MKLDNN=0 USE_ITT=0 BUILD_TEST=0 SKIP_SUBMODULE_CHECK=1 MAX_JOBS=48 CUDA_HOME=/usr/local/corex
export PYTORCH_BUILD_VERSION="2.4.0+corex" PYTORCH_BUILD_NUMBER=1
rm -rf dist
python setup.py bdist_wheel > $HOME/build/wheel24g.log 2>&1
echo wheel_rc=$?
V=$HOME/venvs/torch24test
$V/bin/pip install --no-deps --force-reinstall dist/*.whl > $HOME/build/test24g_install.log 2>&1
echo install_rc=$?
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