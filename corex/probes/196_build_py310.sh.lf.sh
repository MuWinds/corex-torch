set -u
# 目的：用 **py310** 编 fork 的 torch wheel（目标运行时是 py310）。
#
# 关键点：
#   * 把 cp312 的 build 目录挪走，否则 cmake 缓存里全是 3.12 的路径，会编出混血产物；
#   * `LC_ALL=C.UTF-8`：这台机器的 CMake 会按本地 locale 写 CMakeCache.txt，torch 按 UTF-8 读，
#     不指定就会 UnicodeDecodeError（已经踩过三轮）；
#   * `MAX_JOBS=12`：更高并发会偶发 cc1plus 被 Killed。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torch310.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -x "$V/bin/python" ] || { echo "没有 py310 build venv"; exit 66; }
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
if [ -d "$P/build" ] && [ ! -d "$P/build-cp312" ]; then
    mv "$P/build" "$P/build-cp312" && echo "旧 build 目录已挪到 build-cp312"
fi
rm -f "$P/build/CMakeCache.txt" 2>/dev/null || true
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=12 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 60
echo "--- 日志尾 ---"
LC_ALL=C tail -c 500 "$LOG" | tr -d "\r" | tail -4 | cut -c1-140
echo "--- ninja ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "T196_DONE"
