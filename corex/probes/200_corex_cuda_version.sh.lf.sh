set -u
echo "HOST=$(hostname) DATE=$(date -Is)"
ls -d /usr/local/corex* 2>/dev/null
for H in /usr/local/corex/include/cuda.h; do
    [ -f "$H" ] && { echo "== $H"; grep -m3 -n "define CUDA_VERSION" "$H"; }
done
for R in /usr/local/corex/include/cuda_runtime_api.h; do
    [ -f "$R" ] && { echo "== $R"; grep -m3 -n "define CUDART_VERSION" "$R"; }
done
echo "--- nvcc ---"
ls -la /usr/local/corex/bin/nvcc 2>/dev/null || echo "no nvcc"
echo "--- libcudart ---"
ls /usr/local/corex/lib64/libcudart* 2>/dev/null | head -3
echo "T200_DONE"
