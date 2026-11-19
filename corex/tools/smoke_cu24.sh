set -u
hostname
ls -l /usr/local/corex/bin/clang++ /usr/local/corex/bin/clang-16 2>&1
T=$HOME/build/torch24
export PATH=$HOME/venvs/torchbuild310/bin:/usr/local/corex/bin:$PATH
cd "$T/build" || exit 1
OBJ=caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/AbsKernel.cu.o
ninja -v "$OBJ" > /tmp/abs2.log 2>&1
echo "ninja_rc=$?"
grep -m3 "error:" /tmp/abs2.log | cut -c1-200
grep -m1 "clang++" /tmp/abs2.log | cut -c1-80
ls -lh "$OBJ" 2>/dev/null
