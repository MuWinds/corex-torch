#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
cmake .. -DUSE_CUDA=ON -DUSE_ONNX=ON > $HOME/build/reconf24a.log 2>&1
echo cmake_rc=$?
grep -c "TensorFactories.cu" build.ninja
ninja -k 20 > $HOME/build/fix24a.log 2>&1
echo ninja_rc=$?
echo failed=$(grep -c FAILED: $HOME/build/fix24a.log)
grep FAILED: $HOME/build/fix24a.log | sed s/.*torch24..// | sed s/.o..// | sort -u | head -4