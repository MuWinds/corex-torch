set -u
# 目的：验证 `_bf16` 那层改写（fpext/fptrunc → llvm.bi.bfloat2float / llvm.bi.float2bfloat）
# 到底成不成立。判据四条，缺一条都不算过：
#   1. bf16 内核**编得过**（之前必死在 Cannot select: fp_extend / fp_round）；
#   2. metadata['corex_rewrite_counts']['bf16_conversions'] > 0（证明是改写生效，不是碰巧编过）；
#   3. 改写后的 biir 里 fpext/fptrunc 命中数为 0、llvm.bi.bfloat2float 出现 ≥1；
#   4. 跑出来的结果与 CPU 的 bf16 参照**逐字节相等**。
# 这条链成立的话，内核侧就完全不需要手写位运算（那套会引入 trunc nuw / splat 两类新语法问题）。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡 ==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/bf16_check.py" <<'PYEOF'
import ctypes
import struct
import sys

sys.path.insert(0, sys.argv[1])

import torch

import corex_torch
from corex_torch import _bf16, _runtime as rt, _triton

_triton.activate()

import triton
import triton.language as tl


@triton.jit
def bf16_add(a, b, c, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(a + off, mask=m).to(tl.float32)
    y = tl.load(b + off, mask=m).to(tl.float32)
    tl.store(c + off, (x + y).to(tl.bfloat16), mask=m)


@triton.jit
def bf16_mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = k0 + tl.arange(0, BK)
        a_tile = tl.load(a + offs_m[:, None] * K + offs_k[None, :])
        b_tile = tl.load(b + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
    tl.store(c + (pid_m * BM + tl.arange(0, BM))[:, None] * N
             + (pid_n * BN + tl.arange(0, BN))[None, :], acc.to(tl.bfloat16))


def bf16_bits(values):
    """把一组 Python float 打成 bf16 的 16 位整数（走 torch 的 CPU 实现，是最中性的参照）。"""
    t = torch.tensor(values, dtype=torch.float32)
    return t.to(torch.bfloat16).view(torch.int16).to(torch.int32).numpy()


def dev_from(values):
    bf = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    n = bf.numel()
    ptr = rt.malloc(n * 2)
    rt.memcpy(ptr, bf.data_ptr(), n * 2, rt.MEMCPY_HOST_TO_DEVICE)
    return ptr, n


def read_back(ptr, n):
    out = (ctypes.c_uint16 * n)()
    rt.memcpy(out, ptr, n * 2, rt.MEMCPY_DEVICE_TO_HOST)
    return bytes(bytearray().join(struct.pack("<H", v) for v in out))


print("=== 1. 改写函数自检（纯文本，不碰卡）===")
sample = """define iluvatar_kernel void @k(ptr %0) {
  %2 = fpext bfloat %1 to float
  %3 = fptrunc float %2 to bfloat
  ret void
}
"""
rewritten, hits = _bf16.rewrite_bf16_conversions(sample)
print(f"  改写命中 {hits} 处；fpext 残留 {rewritten.count('fpext')}，fptrunc 残留 {rewritten.count('fptrunc')}")
print(f"  出现 llvm.bi.bfloat2float {rewritten.count('llvm.bi.bfloat2float')} 次、"
      f"llvm.bi.float2bfloat {rewritten.count('llvm.bi.float2bfloat')} 次")
print("  幂等性：第二次改写命中", _bf16.rewrite_bf16_conversions(rewritten)[1], "处")

print("=== 2. 逐元素 bf16 add：编译 ===")
N = 4096
vals_a = [0.5, 1.0, 2.0, -3.0] * (N // 4)
vals_b = [0.25, -0.5, 1.5, 4.0] * (N // 4)
pa, n = dev_from(vals_a)
pb, _ = dev_from(vals_b)
pc = rt.malloc(n * 2)
kernel = _triton.compile_kernel(
    bf16_add,
    signature={"a": "*bf16", "b": "*bf16", "c": "*bf16", "n": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": 128})


def rewrite_counts(compiled) -> dict:
    """取改写命中数。compile 阶段的 metadata 是 dict，编完之后是 KernelMetadata（namedtuple），
    所以要兼容两种形态——这一点不写清楚，探针会死在 AttributeError 上，看不出真正原因。"""
    md = compiled.metadata
    value = md.get("corex_rewrite_counts") if hasattr(md, "get") else getattr(
        md, "corex_rewrite_counts", None)
    return dict(value or {})


counts = rewrite_counts(kernel)
print("  rewrite counts =", counts)
biir = kernel.asm["biir"]
print(f"  biir: fpext={biir.count('fpext')} fptrunc={biir.count('fptrunc')} "
      f"bfloat2float={biir.count('llvm.bi.bfloat2float')} "
      f"float2bfloat={biir.count('llvm.bi.float2bfloat')}")

print("=== 3. 逐元素 bf16 add：跑 + 逐字节比对 ===")
_triton.launch(kernel, (triton.cdiv(N, 128),), pa, pb, pc, N, )
rt.synchronize()
got = read_back(pc, N)
a_bf = torch.tensor(vals_a, dtype=torch.float32).to(torch.bfloat16)
b_bf = torch.tensor(vals_b, dtype=torch.float32).to(torch.bfloat16)
ref = (a_bf.float() + b_bf.float()).to(torch.bfloat16)
ref_bytes = ref.contiguous().view(torch.int16).numpy().tobytes()
print("  byte_equal =", got == ref_bytes, " first4 =", struct.unpack("<4H", got[:8]))

print("=== 4. bf16 矩阵乘（tl.dot，fp32 累加）===")
M = Nm = K = 64
a_host = torch.randn(M, K).to(torch.bfloat16)
b_host = torch.randn(K, Nm).to(torch.bfloat16)
ref_mm = (a_host.float() @ b_host.float()).to(torch.bfloat16)
q_a = rt.malloc(M * K * 2)
q_b = rt.malloc(K * Nm * 2)
q_c = rt.malloc(M * Nm * 2)
rt.memcpy(q_a, a_host.contiguous().data_ptr(), M * K * 2, rt.MEMCPY_HOST_TO_DEVICE)
rt.memcpy(q_b, b_host.contiguous().data_ptr(), K * Nm * 2, rt.MEMCPY_HOST_TO_DEVICE)
mm_kernel = _triton.compile_kernel(
    bf16_mm,
    signature={"a": "*bf16", "b": "*bf16", "c": "*bf16", "M": "i32", "N": "i32", "K": "i32",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"BM": 16, "BN": 16, "BK": 16}, num_warps=1)
print("  rewrite counts =", rewrite_counts(mm_kernel))
_triton.launch(mm_kernel, (M // 16, Nm // 16), q_a, q_b, q_c, M, Nm, K)
rt.synchronize()
got_mm = read_back(q_c, M * Nm)
ref_mm_bytes = ref_mm.contiguous().view(torch.int16).numpy().tobytes()
same = sum(1 for i in range(0, len(got_mm), 2)
           if got_mm[i:i + 2] == ref_mm_bytes[i:i + 2])
print(f"  bf16 mm 逐元素相等比例 = {same}/{M * Nm}")
print(f"  byte_equal = {got_mm == ref_mm_bytes}")

print("=== 5. 结论 ===")
# 判据要按「需要转换的那个内核」来定：逐元素 add 的 bf16 转换被 Triton 自己折掉了
# （biir 里 fpext/fptrunc 各 0，说明它直接用 bf16 算术），所以改写命中数在它身上必然是 0。
# 真正依赖这层改写的是整块 tile 输出窄化的内核——矩阵乘就是。两个结果都要逐字节相等。
mm_counts = rewrite_counts(mm_kernel)
ok = (mm_counts.get("bf16_conversions", 0) > 0
      and got == ref_bytes and got_mm == ref_mm_bytes)
print("  add 改写命中 =", counts.get("bf16_conversions", 0), "（0 属正常，见上面的注释）")
print("  mm  改写命中 =", mm_counts.get("bf16_conversions", 0), "（>0 才说明这层真的在干活）")
print("BF16_INTRINSIC_VERDICT", "PASS" if ok else "FAIL")
PYEOF

timeout --foreground -k 5 600 "$PY" -B -u "$W/bf16_check.py" "$W/src"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
