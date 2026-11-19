set -u
echo "=== /usr/local/corex/bin/nvcc 内容 ==="
cat /usr/local/corex/bin/nvcc
echo "=== corex bin 里的 clang ==="
ls -l /usr/local/corex/bin | grep -i 'clang\|llc\|llvm' | head -12
echo "=== 造一个最小 kernel ==="
cat > /tmp/t24.cu <<'EOF'
__global__ void k(float *a) { a[0] = 1.0f; }
EOF
echo "=== 试 ivcore10 ==="
/usr/local/corex/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -c /tmp/t24.cu -o /tmp/t24_iv.o 2>&1 | tail -4
echo "rc=$?"
echo "=== 试 sm_80 ==="
/usr/local/corex/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_80 -c /tmp/t24.cu -o /tmp/t24_sm.o 2>&1 | tail -4
echo "rc=$?"
echo "=== 试默认 arch（不带参数） ==="
/usr/local/corex/bin/clang++ -x cuda --cuda-device-only -c /tmp/t24.cu -o /tmp/t24_def.o 2>&1 | tail -4
echo "rc=$?"
