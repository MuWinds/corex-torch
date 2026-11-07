set -u
# 目的：把「cuda 别名」这版的重编挪到计算节点，并发 32。
# 登录节点 load 18（别人在用），12 并发下 15 分钟只编 100 个文件，按这个速度要两个多小时。
# 计算节点 56 核、卡也是健康的，而 $HOME 共享，直接在那儿编即可。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda5.log
echo "HOST=$(hostname) DATE=$(date -Is)"
rm -f "$P/build/CMakeCache.txt"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=32 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 20
LC_ALL=C tail -c 300 "$LOG" | tr -d "\r" | tail -2 | cut -c1-120
echo "T206_DONE"
