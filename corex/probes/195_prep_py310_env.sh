set -u
# 目的：准备 py310 的构建环境（cp312 的 wheel 已经证明打包链路通，但目标运行时是 py310：
# 厂商 torch 2.1 基线、corex28 环境、以及我们那个 py310 ABI 的 shim 都在 py310 上）。
#
# 依赖必须**在登录节点装**（计算节点没有外网），然后用同一份 $HOME 在计算节点上编。
H=$HOME
PY310=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
V=$H/venvs/torchbuild310
echo "HOST=$(hostname) DATE=$(date -Is)"
"$PY310" --version || exit 66
if [ ! -x "$V/bin/python" ]; then
    "$PY310" -m venv "$V" || exit 66
fi
"$V/bin/pip" install -q -U pip wheel setuptools 2>&1 | tail -1
"$V/bin/pip" install -q "cmake>=3.27" ninja pyyaml typing_extensions filelock jinja2 networkx \
    sympy requests numpy six 2>&1 | tail -2
"$V/bin/python" -c "import sys, cmake, ninja; print('PY', sys.version.split()[0], 'cmake', cmake.__version__)"
echo "PREP_DONE"
