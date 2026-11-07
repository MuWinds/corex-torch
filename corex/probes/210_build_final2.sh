set -u
# 目的：最后一版打包（cuda/corex 别名 + is_cuda + Tensor.cuda + Stream）。
#
# **关键教训**：从 Windows 打包上传时，tar 保留的是 Windows 的 mtime，可能比节点上的 .o 还旧，
# ninja 于是认为「不用重编」，wheel 里还是旧二进制（前面 19:19 那次 2 分钟就"编完"就是这个原因）。
# 所以装完必须 `touch` 改动过的 C++ 文件，逼 ninja 重编。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda7.log
echo "HOST=$(hostname) DATE=$(date -Is)"
( cd "$H/work" && tar xzf cudamod4.tgz -C "$P" ) || exit 1
touch "$P/c10/core/Device.cpp" "$P/c10/core/Device.h" "$P/c10/core/DeviceType.cpp" "$P/c10/core/TensorImpl.h"
echo "校验：Device.cpp corex 别名=$(grep -c '"corex"' "$P/c10/core/Device.cpp")  TensorImpl is_cuda 补丁=$(grep -c kPrivateUse1 "$P/c10/core/TensorImpl.h")  Stream 包装=$(grep -c 'def Stream' "$P/torch/cuda/__init__.py")"
rm -f "$P/build/CMakeCache.txt"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=24 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 15
echo "已启动"; echo "T210_DONE"
