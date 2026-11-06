set -u
# 目的（承接 011）：011 已经确认厂商 torch 走的是 intrinsic 分支——
#   `__CUDA_ARCH__ 300`、`__ILUVATAR__ 1`，于是 BFloat16-inl.h 里
#   `(__CUDA_ARCH__ >= 800 || defined(__ILUVATAR__))` 为真，bf16 转换调 `__float2bfloat16` /
#   `__bfloat162float`。这两个函数的**实现**在 `$CX/include/cuda_bf16.hpp`（427 / 543 行）。
# 这一份探针把那段实现读出来，并把它与我们自己写位运算的版本编成 IR 对照：
#   若厂商实现也是位运算 → 我们的路线与厂商完全一致（可以直接照抄其 op 序列）；
#   若厂商实现是某个 intrinsic → 我们该改发那个 intrinsic，而不是位运算。
CX=/usr/local/corex
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct/bf16
export PATH=$CX/bin:$PATH
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

echo "=== 1. 厂商 cuda_bf16.hpp 里两个转换的实现（427 与 543 行附近）==="
sed -n '420,470p' $CX/include/cuda_bf16.hpp
echo "--------"
sed -n '535,575p' $CX/include/cuda_bf16.hpp

echo "=== 2. 用厂商头编一个只做转换的 kernel，看设备 IR ==="
cat > "$W/conv2.cu" <<'CUEOF'
#include <cuda_bf16.h>
extern "C" __global__ void k_conv(const __nv_bfloat16 *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = __bfloat162float(in[i]);      // 展宽
    f = f * 2.0f + 1.0f;
    out[i] = f;
    out[i + n] = __bfloat162float(__float2bfloat16(f));   // 收窄再展宽（往返）
}
CUEOF
$CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -O2 -S -emit-llvm "$W/conv2.cu" -o "$W/conv2.ll" 2>&1 | head -8
echo "--- conv2.ll 关键行（fpext/fptrunc 有没有出现）---"
grep -cE "fpext|fptrunc" "$W/conv2.ll" 2>/dev/null | sed 's/^/  fpext+fptrunc 命中数: /'
grep -nE "fpext|fptrunc|shl |lshr |and i32|bitcast|llvm\.bi\.|call " "$W/conv2.ll" 2>/dev/null | head -20

echo "=== 3. 同一段代码用「纯位运算」写（对照）==="
cat > "$W/bit2.cu" <<'CUEOF'
#include <cstdint>
__device__ __forceinline__ float bf2f(unsigned short u) {
    unsigned int v = (unsigned int)u << 16;
    float f; __builtin_memcpy(&f, &v, 4); return f;
}
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int v; __builtin_memcpy(&v, &f, 4);
    v += 0x7fffu + ((v >> 16) & 1u);
    return (unsigned short)(v >> 16);
}
extern "C" __global__ void k_bit(const unsigned short *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = bf2f(in[i]);
    f = f * 2.0f + 1.0f;
    out[i] = f;
    out[i + n] = bf2f(f2bf(f));
}
CUEOF
$CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -O2 -S -emit-llvm "$W/bit2.cu" -o "$W/bit2.ll" 2>&1 | head -8
echo "--- bit2.ll 关键行 ---"
grep -cE "fpext|fptrunc" "$W/bit2.ll" 2>/dev/null | sed 's/^/  fpext+fptrunc 命中数: /'
grep -nE "shl |lshr |and i32|bitcast" "$W/bit2.ll" 2>/dev/null | head -12

echo "=== 4. 两份 IR 都过一遍 llc，确认都能出机器码 ==="
for f in conv2 bit2; do
    if [ -f "$W/$f.ll" ]; then
        if $CX/bin/llc -march=bi -mcpu=ivcore10 -mtriple=bi-iluvatar-ilurt \
             -filetype=obj "$W/$f.ll" -o "$W/$f.o" 2>"$W/$f.err"; then
            echo "  $f: llc OK ($(stat -c%s "$W/$f.o" 2>/dev/null) B)"
        else
            echo "  $f: llc 失败 -> $(head -2 "$W/$f.err" | tr '\n' ' ')"
        fi
    fi
done

echo "=== 5. 结论行（给文档抄）==="
V=$(grep -cE "^.*fpext|fptrunc" "$W/conv2.ll" 2>/dev/null || echo "?")
echo "  厂商 intrinsic 路径 IR 里 fpext/fptrunc 计数 = ${V:-0}"
echo "  位运算路径 IR 里 fpext/fptrunc 计数 = $(grep -cE 'fpext|fptrunc' "$W/bit2.ll" 2>/dev/null || echo 0)"
echo "BF16_VENDOR_DONE"
