set -u
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torchcuda9.log
echo "HOST=$(hostname) DATE=$(date -Is)"
( cd "$H/work" && tar xzf cudamod6.tgz -C "$P" ) || exit 1
echo "校验：current_stream 用 _runtime=$(grep -c 'return _runtime.Stream.default()' "$P/torch/corex/__init__.py")"
rm -f "$P/build/CMakeCache.txt"
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=24 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 12
echo "已启动"; echo "T213_DONE"
