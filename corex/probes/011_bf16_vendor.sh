set -u
# 目的：查清「厂商 torch 2.1 的 bf16 是怎么在这张卡上跑起来的」，从而决定我们内核里该发什么指令。
#
# 为什么这件事能定性：ATen 里 bf16 的全部算术都是
#     operator+(BFloat16 a, b) { return static_cast<float>(a) + static_cast<float>(b); }
# 也就是说「bf16 算子」在 NVIDIA 上本质是「fp32 算 + 两次转换」。厂商 torch 在这张卡上跑通了
# bf16（沙箱探卡用的就是 8x8 bf16 matmul），说明它那两次转换在这张卡上有指令可用。
# 参照物是 `torch/include/c10/util/BFloat16-inl.h`：
#     BFloat16(float v) : x(__bfloat16_as_ushort(__float2bfloat16(v)));   // 定义了 __ILUVATAR__
#     operator float()  : return __bfloat162float(...);                   // 同上
# 所以真问题是：**`__float2bfloat16` / `__bfloat162float` 在 CoreX 上落到哪条指令**？
# 三种可能：位运算（我们的路子）、专用 intrinsic、或者 llc 能选的 fp 转换（那就是 Triton 的
# lowering 选错了 IR 形态，我们照着厂商的写法发同样的东西即可）。
#
# 做法：直接用厂商 clang 编一个只做两种转换的 kernel，dump 设备 IR 与 SASS 级汇编。
CX=/usr/local/corex
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct/bf16
export PATH=$CX/bin:$PATH
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

echo "=== 1. 厂商头文件里 __float2bfloat16 / __bfloat162float 的实现 ==="
for h in cuda_bf16.h cuda_bf16.hpp; do
    if [ -e "$CX/include/$h" ]; then
        echo "--- $CX/include/$h ---"
        grep -n "__float2bfloat16\b\|__bfloat162float\b" "$CX/include/$h" | head -12
    else
        echo "--- $CX/include/$h 不存在 ---"
    fi
done
echo "--- 头文件里出现的 bf16 相关 intrinsic 名（去重）---"
grep -rhoE "__nv_[a-z0-9_]*bfloat16[a-z0-9_]*|__bi_[a-z0-9_]*" $CX/include/cuda_bf16.h "$CX/include/cuda_bf16.hpp" 2>/dev/null | sort -u | head -20

echo "=== 2. __CUDA_ARCH__ 与 __ILUVATAR__ 到底是多少（决定 ATen 走哪条分支）==="
cat > "$W/macros.cu" <<'CUEOF'
__global__ void k_macros(int *out) {
#if defined(__CUDA_ARCH__)
    out[0] = __CUDA_ARCH__;
#else
    out[0] = -1;
#endif
#ifdef __ILUVATAR__
    out[1] = 1;
#else
    out[1] = 0;
#endif
}
CUEOF
$CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -O2 -S -emit-llvm "$W/macros.cu" -o "$W/macros.ll" 2>&1 | head -5
grep -n "__CUDA_ARCH__" "$W/macros.ll" 2>/dev/null | head -3
echo "--- 直接问预处理宏 ---"
echo "" | $CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -dM -E - 2>/dev/null | grep -E "ILUVATAR|CUDA_ARCH" | head -5

echo "=== 3. 两种 bf16 转换编出来的设备 IR ==="
cat > "$W/conv.cu" <<'CUEOF'
#include <cstdint>
__global__ void k_conv(const unsigned short *in, float *out, float *fin, unsigned short *fout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // 展宽：bf16 -> f32（走厂商头的 intrinsic，不是我们自己写位运算）
    __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16 *>(in + i);
    out[i] = __bfloat162float(b);
    // 收窄：f32 -> bf16
    out[i] = out[i] + 1.0f;
    fout[i] = __bfloat16_as_ushort(__float2bfloat16(fin[i]));
}
CUEOF
$CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -O2 -S -emit-llvm "$W/conv.cu" -o "$W/conv.ll" 2>&1 | head -8
echo "--- conv.ll 里的关键行 ---"
grep -nE "fpext|fptrunc|bitcast|shl|ashr|lshr|and |or |llvm\.bi\.|intrinsic" "$W/conv.ll" 2>/dev/null | head -25
echo "--- 若上面为空，说明全被折成了别的形态，把函数体打出来看 ---"
sed -n "/define .*k_conv/,/^}/p" "$W/conv.ll" 2>/dev/null | head -40

echo "=== 4. 编成目标文件后看真实指令（llc）==="
$CX/bin/llc -march=bi -mcpu=ivcore10 -mtriple=bi-iluvatar-ilurt -filetype=asm "$W/conv.ll" -o "$W/conv.s" 2>&1 | head -5
echo "--- conv.s 里的转换相关指令 ---"
grep -nE "cvt|bf16|shl|shr|and|lshr" "$W/conv.s" 2>/dev/null | head -20

echo "=== 5. 我们自己用位运算写的版本（对照）==="
cat > "$W/bit.cu" <<'CUEOF'
#include <cstdint>
__device__ __forceinline__ float bf16_to_f32_bits(unsigned short u) {
    unsigned int v = (unsigned int)u << 16;
    float f;
    __builtin_memcpy(&f, &v, 4);
    return f;
}
__device__ __forceinline__ unsigned short f32_to_bf16_bits(float f) {
    unsigned int v;
    __builtin_memcpy(&v, &f, 4);
    unsigned int lsb = (v >> 16) & 1u;
    v += 0x7fffu + lsb;                      // RNE
    return (unsigned short)(v >> 16);
}
__global__ void k_bit(const unsigned short *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf16_to_f32_bits(in[i]);
    out[i] = out[i] + 1.0f;
    out[i] = bf16_to_f32_bits(f32_to_bf16_bits(out[i]));
}
CUEOF
$CX/bin/clang++ -x cuda --cuda-device-only --cuda-gpu-arch=ivcore10 -nocudalib \
    -I$CX/include -O2 -S -emit-llvm "$W/bit.cu" -o "$W/bit.ll" 2>&1 | head -8
grep -nE "fpext|fptrunc|shl|ashr|lshr|and |or " "$W/bit.ll" 2>/dev/null | head -20
echo "BIT_IR_DONE"
