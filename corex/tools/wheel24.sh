#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24 || exit 1
export USE_CUDA=1 USE_CUDNN=0 USE_KINETO=0 USE_NCCL=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_NNPACK=0 USE_XNNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
export USE_FBGEMM=0 USE_MKLDNN=0 USE_ITT=0 BUILD_TEST=0 SKIP_SUBMODULE_CHECK=1 MAX_JOBS=48 CUDA_HOME=/usr/local/corex
export TORCH_BUILD_VERSION="2.4.0+corex" TORCH_BUILD_NUMBER=1
python setup.py bdist_wheel > $HOME/build/wheel24.log 2>&1
echo wheel_rc=$?
ls -la dist/*.whl 2>/dev/null | tail -2
tail -5 $HOME/build/wheel24.log