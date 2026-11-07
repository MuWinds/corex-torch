set -u
# 目的：在计算节点上后台编 fork 的 torch（CPU-only）。登录节点被别人压满（10 分钟只编 4 个文件），
# 计算节点的 CPU 是空的，而 $HOME 共享，所以只需把 build 挪过去。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild
LOG=$H/work/torchbuild6.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -d "$P" ] || { echo "没有源码树"; exit 66; }
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C LANG=C PATH=$V/bin:$PATH MAX_JOBS=44 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 40
echo "--- 日志尾 ---"
LC_ALL=C tail -c 600 "$LOG" | tr -d "\r" | tail -4
echo "--- ninja ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "BUILD_ON_NODE_STARTED"
