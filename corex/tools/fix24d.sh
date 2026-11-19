#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
rm -f caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.*.o
ninja -k 20 > $HOME/build/fix24d.log 2>&1
echo ninja_rc=$? failed=$(grep -c FAILED: $HOME/build/fix24d.log)
F=$(find . -path "*nested/cuda*" -name "NestedTensorMatmul*.o" | head -1)
echo obj=$F
nm -C "$F" 2>/dev/null | grep -c bmm_nested