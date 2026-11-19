#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
cmake .. -DUSE_CUDA=ON -DUSE_ONNX=ON > $HOME/build/reconf24r.log 2>&1
echo CFG=$?
ninja -k 50 > $HOME/build/scan24k.log 2>&1
echo ninja_rc=$?
echo failed=$(grep -c FAILED: $HOME/build/scan24k.log)
grep FAILED: $HOME/build/scan24k.log | sed s/.*torch24..// | sed s/.o..// | sort -u | head -10