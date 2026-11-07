set -u
# 目的：在远端把 **fork 的 torch 源码**真编一次（CPU-only），拿到 `import torch` 自带
# corex 后端的 wheel。这一步是「fork 形态」的硬证据：之前一直是把 torch/corex 覆盖进
# 已装 wheel 里，形态对但不是真编出来的。
#
# 为什么后台起：全量编译 30–60 分钟，队列单脚本上限 1800 s，前台必被杀。
#
# 关键约束：
#   * torch 2.8 要求 CMake >= 3.27，系统只有 3.26.4 → 用 pip 装一份到 build venv；
#   * 只编 CPU（USE_CUDA=0），不需要卡片参与，登录节点就能编；
#   * 源码用 `python setup.py bdist_wheel`，产物进 dist/，之后可以被任何 venv 安装。
H=$HOME
SRC=$H/work/torchfork/corex-torch
VENV=$H/venvs/torchbuild
LOG=$H/work/torchbuild.log
echo "HOST=$(hostname) DATE=$(date -Is)"

if [ ! -x "$VENV/bin/python" ]; then
    echo "=== 建 build venv ==="
    python3 -m venv "$VENV" || exit 1
fi
export PATH=$VENV/bin:$PATH
echo "=== 装构建依赖 ==="
"$VENV/bin/pip" install -q -U pip wheel setuptools 2>&1 | tail -2
"$VENV/bin/pip" install -q "cmake>=3.27" ninja pyyaml typing_extensions filelock jinja2 networkx \
    sympy requests numpy 2>&1 | tail -3
"$VENV/bin/cmake" --version | head -1
"$VENV/bin/ninja" --version

if [ ! -d "$SRC" ]; then
    echo "=== 解包源码 ==="
    mkdir -p "$H/work/torchfork"
    ( cd "$H/work/torchfork" && tar xzf "$H/work/torchfork.tgz" ) || exit 1
fi
cd "$SRC" || exit 1
echo "源码 $SRC（$(ls | wc -l) 个顶层条目）"

if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then
    echo "ALREADY_BUILDING"
    exit 0
fi

: > "$LOG"
( cd "$SRC" && MAX_JOBS=48 setsid nohup "$VENV/bin/python" setup.py bdist_wheel \
      >> "$LOG" 2>&1 & )
sleep 30
echo "--- 日志头 ---"
tail -8 "$LOG"
echo "--- 编译进程 ---"
pgrep -c ninja 2>/dev/null || echo 0
echo "TORCH_BUILD_STARTED"
