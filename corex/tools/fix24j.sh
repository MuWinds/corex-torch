#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
rm -f caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/TensorTopK.*.o
ninja -k 20 > $HOME/build/fix24j.log 2>&1
echo ninja_rc=$? failed=$(grep -c FAILED: $HOME/build/fix24j.log)
grep FAILED: $HOME/build/fix24j.log | head -3