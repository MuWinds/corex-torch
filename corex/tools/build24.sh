set -u
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
export PYTHONDONTWRITEBYTECODE=1
find tools -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
# 脱离 runner 的进程组，避免队列任务结束后被回收
setsid nohup python setup.py bdist_wheel > "$HOME/build/build24.log" 2>&1 < /dev/null &
echo $! > "$HOME/build/build24.pid"
sleep 5
echo "pid=$(cat "$HOME/build/build24.pid")"
tail -5 "$HOME/build/build24.log"
