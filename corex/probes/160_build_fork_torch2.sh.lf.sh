set -u
# 目的：编 fork 的 torch（同 159，但**不在计算节点装依赖**——gpu-node 没有外网，
# pip 会一直重试然后失败）。依赖已经在登录节点装进 ~/venvs/torchbuild，这里直接用。
H=$HOME
SRC=$H/work/torchfork/corex-torch
VENV=$H/venvs/torchbuild
LOG=$H/work/torchbuild.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -x "$VENV/bin/python" ] || { echo "没有 build venv"; exit 66; }
"$VENV/bin/cmake" --version | head -1

if [ ! -d "$SRC" ]; then
    mkdir -p "$H/work/torchfork"
    ( cd "$H/work/torchfork" && tar xzf "$H/work/torchfork.tgz" ) || exit 1
fi
cd "$SRC" || exit 1
echo "源码 $SRC"

if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then
    echo "ALREADY_BUILDING"; exit 0
fi

: > "$LOG"
( cd "$SRC" && MAX_JOBS=48 setsid nohup "$VENV/bin/python" setup.py bdist_wheel \
      >> "$LOG" 2>&1 & )
sleep 40
echo "--- 日志尾 ---"
tail -10 "$LOG" | cut -c1-160
echo "--- ninja 进程 ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "TORCH_BUILD_STARTED"
