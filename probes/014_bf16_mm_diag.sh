set -u
# 目的：bf16 矩阵乘的输出窄化编不过（`Cannot select: bf16 = fp_round`），但逐元素 add 能过。
# 说明「整块 .to(tl.bfloat16)」发出的不是标量 fptrunc，而我的改写只认标量形态。
# 这份探针只做一件事：把 mm 内核进 llc 之前的 biir 落盘，把里面所有涉及 bfloat 的转换行
# 打出来，再据此决定改写要覆盖哪些形态（向量 / 不同拼写 / 不同指令名）。
PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
export COREX_BF16_DUMP=$W/mm_biir.ll
echo "HOST=$(hostname) DATE=$(date -Is) DUMP=$COREX_BF16_DUMP"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/mm_diag.py" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])

import corex_torch
from corex_torch import _triton

_triton.activate()

import triton
import triton.language as tl


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


try:
    _triton.compile_kernel(
        bf16_mm,
        signature={"a": "*bf16", "b": "*bf16", "c": "*bf16", "M": "i32", "N": "i32", "K": "i32",
                   "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
        constexprs={"BM": 16, "BN": 16, "BK": 16}, num_warps=1)
    print("COMPILE_OK")
except Exception as exc:                                # noqa: BLE001
    print("COMPILE_FAIL", type(exc).__name__, str(exc).splitlines()[0][:200])
print("DIAG_DONE")
PYEOF

timeout --foreground -k 5 300 "$PY" -B -u "$W/mm_diag.py" "$W/src"
echo "RC=$?"

echo "=== 1. 落盘的 biir 里所有涉及 bfloat 的行 ==="
if [ -f "$COREX_BF16_DUMP" ]; then
    wc -l "$COREX_BF16_DUMP" | sed 's/^/  行数: /'
    grep -n "bfloat" "$COREX_BF16_DUMP" | head -30
    echo "--- 转换类指令的形态统计（不限 bfloat）---"
    grep -oE "fpext [^%]*%[^ ]+ to [a-z0-9 <>x]+|fptrunc [^%]*%[^ ]+ to [a-z0-9 <>x]+" "$COREX_BF16_DUMP" | sed 's/%[^ ]*//' | sort | uniq -c | head -12
    echo "--- 声明区里的 bi intrinsic ---"
    grep -n "^declare.*llvm.bi" "$COREX_BF16_DUMP" | head -10
else
    echo "  没有落盘文件：改写函数可能没被调用（说明编到 llir 阶段就失败了）"
fi
echo "DIAG_ALL_DONE"
