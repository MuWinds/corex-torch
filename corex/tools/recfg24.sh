#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
cmake .. -DUSE_CUDA=ON -DUSE_ONNX=ON > $HOME/build/reconf24o.log 2>&1
echo CFG=$?
grep -m1 USE_ONNX:BOOL CMakeCache.txt
echo CUOBJ=$(grep -c .cu.o build.ninja)
grep -c onnx build.ninja
grep -n -A3 CMake\ Error $HOME/build/reconf24o.log | head -8 | cut -c1-140