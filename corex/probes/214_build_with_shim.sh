set -u
# 目的：让 **wheel 自带 torch/corex/_corex_shim.so**，装完不用再编。
#   做法：torch/CMakeLists.txt 里加 corex_shim 目标（MODULE，输出名 _corex_shim），
#   install 到 ${CMAKE_INSTALL_PREFIX}/corex —— setup.py 把前缀指向 <build>/torch，
#   所以正好落在 wheel 的 torch/corex/ 下。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchshim.log
echo "HOST=$(hostname) DATE=$(date -Is)"
( cd "$H/work" && tar xzf shimmod.tgz -C "$P" ) || exit 1
touch "$P/torch/CMakeLists.txt"
echo "源码在树里=$(ls -la "$P/corex/src/csrc/corex_shim.cpp" | awk '{print $5}') 字节；CMake 目标=$(grep -c 'add_library(corex_shim' "$P/torch/CMakeLists.txt")"
rm -f "$P/build/CMakeCache.txt"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=24 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 15
echo "已启动"; echo "T214_DONE"
