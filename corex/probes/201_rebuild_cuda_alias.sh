set -u
# 目的：把「cuda 设备名别名」的改动装进源码树并增量重编（只改了 2 个 C++ 文件 + 几个 .py，
# 所以是增量：编这几个 + 重链）。
#
# 为什么在登录节点编：沙箱队列被别人 1800s 级的脚本占住了；而这次只有两个 .cpp，几分钟的事。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -d "$P" ] || { echo "没有源码树"; exit 66; }
( cd "$H/work" && tar xzf cudamod.tgz -C "$P" ) || exit 1
echo "装好后 Device.cpp 里 cuda 命中=$(grep -c 'PrivateUse1' "$P/c10/core/Device.cpp")"
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
rm -f "$P/build/CMakeCache.txt" 2>/dev/null || true
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=12 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 45
echo "--- 日志尾 ---"
LC_ALL=C tail -c 500 "$LOG" | tr -d "\r" | tail -4 | cut -c1-140
echo "T201_DONE"
