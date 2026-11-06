set -u
# 目的：corex-torch 要自己当「设备后端」，所以先把手头这套厂商运行时到底有什么点清：
#   1. /usr/local/corex 下有哪些共享库（soname 决定 ctypes 怎么开、版本号决定能声明成什么）；
#   2. 兼容层自称的 CUDA / cuBLAS / cuDNN 版本（决定上游代码里哪些 API 不能碰）；
#   3. 有没有 cublasLt / cusparse / nccl —— 决定将来算子能不能借厂商库，还是只能走 Triton；
#   4. 编译器与 gcc（PrivateUse1 的分配器需要一个小 C++ 扩展，要提前确认能编）。
# 这段不碰卡，只读文件与头文件，但仍然要在计算节点跑：登录节点没有 /usr/local/corex。
C=/usr/local/corex
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 1. corex 根目录与版本 ==="
ls -ld $C 2>&1
ls -d /usr/local/corex-* 2>&1
echo "--- bin 里的关键工具 ---"
for t in clang++ clang llc lld ld.lld fatbinary nvcc ixsmi; do
    if [ -x "$C/bin/$t" ]; then echo "  $t: yes"; else echo "  $t: no"; fi
done
echo "--- clang 版本 ---"
$C/bin/clang++ --version 2>&1 | head -2

echo "=== 2. lib64 里的运行时库（soname 与大小）==="
for lib in libcuda.so libcuda.so.1 libcudart.so libcudart.so.10.2 libcublas.so libcublasLt.so \
           libcudnn.so libcufft.so.10 libcurand.so.10 libcusolver.so.10 libcusparse.so.10 libnccl.so; do
    if [ -e "$C/lib64/$lib" ]; then
        printf '  %-22s -> %s\n' "$lib" "$(readlink -f "$C/lib64/$lib" 2>/dev/null)"
    fi
done
echo "--- lib64 全部（前 40 个）---"
ls $C/lib64 2>/dev/null | head -40

echo "=== 3. 兼容层自称的版本 ==="
grep -rhn "define CUDART_VERSION" $C/include/ 2>/dev/null | head -3
grep -rhn "define CUDA_VERSION" $C/include/cuda.h 2>/dev/null | head -3
for h in cublas_v2.h cublasLt.h cudnn_version.h cudnn.h curand.h cufft.h nccl.h; do
    if [ -e "$C/include/$h" ]; then
        echo "--- $h ---"
        grep -hoE "define (CUBLAS_VER_(MAJOR|MINOR|PATCH)|CUDNN_(MAJOR|MINOR|PATCHLEVEL)|CUFFT_VER_(MAJOR|MINOR)|NCLL_MAJOR)[^/]*" "$C/include/$h" 2>/dev/null | head -6
    fi
done

echo "=== 4. 编译器（PrivateUse1 的 C++ 扩展要用）==="
for cc in gcc g++ /data/software/gcc/bin/gcc; do
    command -v $cc >/dev/null 2>&1 && echo "  $cc: $($cc --version 2>&1 | head -1)"
done
ls -d /data/software/gcc* 2>/dev/null | head -5
echo "--- cmake ---"
command -v cmake >/dev/null 2>&1 && cmake --version 2>&1 | head -1

echo "=== 5. 驱动侧（设备节点与库版本）==="
ls /dev | grep -iE 'iluvatar|ivcore|corex|bi' 2>/dev/null | head -10
lsmod 2>/dev/null | grep -iE 'ivcore|iluvatar|corex' | head -5
cat /proc/driver/ivcore/version 2>/dev/null || cat /sys/module/ivcore/version 2>/dev/null || echo "(无 ivcore version 文件)"
echo "SURVEY_DONE"
