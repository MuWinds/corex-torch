set -u
# 目的：把**当前** torch/corex 更新进源码树，重新打 wheel（只改了 .py，cmake 无活可干，
# 所以这一步只花打包时间），然后重装验证。
H=$HOME
P=$H/work/torchfork2/pytorch
V=$H/venvs/torchbuild310
LOG=$H/work/torch310b.log
echo "HOST=$(hostname) DATE=$(date -Is)"
[ -d "$P/torch/corex" ] || { echo "源码树里没有 torch/corex"; exit 66; }
rm -rf "$P/torch/corex"; mkdir -p "$P/torch/corex"
tar xzf "$H/work/tfinal.tgz" -C "$P/torch/corex" --strip-components=2 || exit 1
echo "更新后文件数=$(ls "$P/torch/corex" | wc -l)，sdpa 实现=$(grep -c _sdpa_impl "$P/torch/corex/_aten_kernels.py")"
if pgrep -f "setup.py bdist_wheel" >/dev/null 2>&1; then echo "ALREADY_BUILDING"; exit 0; fi
rm -f "$P/dist"/*.whl 2>/dev/null || true
cd "$P" || exit 66
: > "$LOG"
( cd "$P" && setsid nohup env LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=$V/bin:$PATH MAX_JOBS=12 \
    USE_NCCL=0 USE_DISTRIBUTED=0 USE_CUDA=0 USE_ROCM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_KINETO=0 \
    BUILD_TEST=0 $V/bin/python setup.py bdist_wheel >> "$LOG" 2>&1 & )
sleep 60
echo "--- 日志尾 ---"
LC_ALL=C tail -c 400 "$LOG" | tr -d "\r" | tail -3 | cut -c1-130
ls -la "$P/dist" 2>/dev/null | tail -2
echo "T199_DONE"
