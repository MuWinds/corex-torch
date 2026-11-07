set -u
# 目的：把「cuda 与 corex 两个别名 + Tensor.cuda 补丁」一次性装进源码树并重编打包。
#   - Device.cpp 改了 → 重编 1 个文件 + 重链（几分钟）；
#   - 其余是 .py，不进编译，只影响打包。
# 队列现在落在 gpu-node-06（卡是死的），但编译不需要卡，正好用它的 CPU。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda6.log
echo "HOST=$(hostname) DATE=$(date -Is)"
( cd "$H/work" && tar xzf cudamod3.tgz -C "$P" ) || exit 1
echo "别名检查：Device.cpp 里 corex=$(grep -c '"corex"' "$P/c10/core/Device.cpp") cuda=$(grep -c '"cuda"' "$P/c10/core/Device.cpp")"
echo "Tensor.cuda 补丁=$(grep -c '_tensor_cuda' "$P/torch/cuda/__init__.py")"
rm -f "$P/build/CMakeCache.txt"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=24 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 15
echo "已启动"; echo "T209_DONE"
