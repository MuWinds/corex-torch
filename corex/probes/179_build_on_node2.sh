set -u
# 目的：在计算节点后台编 fork 的 torch（CPU-only）。
#
# 两个已踩过的坑写在这里：
#   1. 登录节点被别人压满（10 分钟只编 4 个文件）→ 编译放计算节点，$HOME 共享，不用搬数据；
#   2. `CMakeCache.txt` 里可能残留 **GBK** 字节（之前用中文 locale 配置过），torch 的
#      `setup_helpers/cmake_utils.py` 按 UTF-8 读会抛 UnicodeDecodeError → 每次启动前删掉缓存，
#      并用 LC_ALL=C 让新的缓存是纯 ASCII。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild
LOG=$H/work/torchbuild7.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -d "$P" ] || { echo "没有源码树"; exit 66; }
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
rm -f "$P/build/CMakeCache.txt"
rm -rf "$P/build/.cmake"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C LANG=C PATH=$V/bin:$PATH MAX_JOBS=44 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 60
echo "--- 日志尾 ---"
LC_ALL=C tail -c 700 "$LOG" | tr -d "\r" | tail -5
echo "--- ninja ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "BUILD_ON_NODE_STARTED"
