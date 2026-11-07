set -u
# 目的：接着编/打包 fork 的 torch。上一次在 153 个待编目标里推进到 12 个时，
# 有 cc1plus 被 `Killed`（无内存/无 ulimit 上限，怀疑是瞬间竞争）——把并发降到 12 重试。
# ninja 会从上次的进度接着来，不会重头编。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild
LOG=$H/work/torchbuild9.log
echo "HOST=$(hostname) DATE=$(date -Is)"
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
rm -f "$P/build/CMakeCache.txt.bak"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C LANG=C PATH=$V/bin:$PATH MAX_JOBS=12 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 50
echo "--- 日志尾 ---"
LC_ALL=C tail -c 500 "$LOG" | tr -d "\r" | tail -4 | cut -c1-140
echo "--- ninja ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "T190_DONE"
