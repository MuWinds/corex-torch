set -u
# 目的：在计算节点上对 torch 2.4 做 cmake-only 配置冒烟。
# 关键点：CUDA 不走 CUDA 语言，走 clang++ -x cuda；nvcc 完全不参与。
T=$HOME/build/torch24
export CUDA_HOME=/usr/local/corex
export PATH=$HOME/venvs/torchbuild310/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export USE_CUDA=1
export USE_CUDNN=0
export USE_KINETO=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_TENSORPIPE=0
export USE_GLOO=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_NNPACK=0
export USE_XNNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_FBGEMM=0
export USE_MKLDNN=0
export USE_ONNX=0
export USE_ITT=0
export BUILD_TEST=0
export SKIP_SUBMODULE_CHECK=1
export MAX_JOBS=48

cd "$T" || exit 1

# 每次冒烟都从干净 cache 开始，避免旧的 CUDA_NVCC_EXECUTABLE / 假 toolkit 根残留
rm -rf "$T/build"

echo "--- toolchain ---"
/usr/local/corex/bin/clang++ --version | head -2
python -c 'import sys; print(sys.version)'
python -c 'import cmake; print("cmake", cmake.__version__)'
echo "--- launcher smoke ---"
printf '__global__ void k(float*a){a[0]=1.f;}\n' > /tmp/k24.cu
bash cmake/public/corex-cxx-launcher.sh /usr/bin/c++ -c /tmp/k24.cu -o /tmp/k24.o && echo LAUNCHER_OK
echo "--- configure ---"
python setup.py build --cmake-only > "$HOME/build/cfg24.log" 2>&1
rc=$?
tail -50 "$HOME/build/cfg24.log"
exit $rc
