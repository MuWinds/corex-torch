set -u
T=$HOME/build/torch24
export CUDA_HOME=/usr/local/corex
export PATH=$HOME/venvs/torchbuild310/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export MAX_JOBS=48
cd "$T/build" || exit 1
# 编译选项变了，先刷新生成文件（保留已有对象，不清 build）
cmake . > "$HOME/build/reconf24.log" 2>&1 || { tail -20 "$HOME/build/reconf24.log"; exit 1; }
ninja -k 50 > "$HOME/build/scan24.log" 2>&1
echo "ninja_rc=$?"
echo "failed_count=$(grep -c 'FAILED:' "$HOME/build/scan24.log")"
grep -m 14 -A 3 "error:" "$HOME/build/scan24.log" | cut -c1-190
