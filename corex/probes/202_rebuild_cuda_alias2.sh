set -u
# 目的：装「cuda 设备名别名」整组改动并重编。
#   改动面：c10/core/Device.{h,cpp}（"cuda" 解析到 PrivateUse1 + is_cuda 为真）、
#           c10/core/DeviceType.cpp（放开 "cuda" 保留名）、
#           torch/cuda/__init__.py（委托）、torch/backends/cuda/__init__.py（is_built）、
#           torch/__init__.py（version.cuda）、torch/corex/*（DEVICE_NAME="cuda"）。
#   改动里有头文件（Device.h），所以会大面积重编，约 40–60 分钟。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda2.log
echo "HOST=$(hostname) DATE=$(date -Is)"
pkill -f "setup.py bdist_wheel" 2>/dev/null || true
sleep 3
( cd "$H/work" && tar xzf cudamod2.tgz -C "$P" ) || exit 1
echo "校验：Device.h is_cuda 补丁=$(grep -c 'PrivateUse1;' "$P/c10/core/Device.h")  Device.cpp cuda 别名=$(grep -c 'PrivateUse1' "$P/c10/core/Device.cpp")  cuda/__init__ 委托=$(grep -c 'corex fork' "$P/torch/cuda/__init__.py")"
rm -f "$P/build/CMakeCache.txt" 2>/dev/null || true
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=12 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 45
echo "--- 日志尾 ---"
LC_ALL=C tail -c 400 "$LOG" | tr -d "\r" | tail -3 | cut -c1-130
echo "T202_DONE"
