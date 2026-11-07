set -u
# 探针 123：aten::mm 走 C++ 之后，为什么**非方阵**算出垃圾？（形状矩阵 + 逐元素对 CPU）
#
# 背景：probes/009 实测 `mm 64x128 @ 128x256` max_rel=2.4e+04，而 256³ 看起来正常。
# 教训（Lead 定的口径）：**「方阵 + 只报吞吐」的验收会整类漏掉 grid/参数错误**——方阵上
# X/Y 交换不可见，吞吐又和正确性无关。所以本探针：
#   1. 按形状矩阵覆盖 方阵 / 非方阵 / 非整除 / 极窄 五种形态；
#   2. 每个形状都**与 CPU 逐元素比对**（报 max_abs_diff 与 max_rel），不看吞吐；
#   3. aten 路径与 `shim.launch` 直调路径**分别**测（两条路共用同一个 handle，
#      若错因在参数/网格，两条应当一起错）；
#   4. 每个用例前先把输出所在的**缓存池块污染**（填哨兵再释放），这样「没写完」的区域
#      会留下哨兵值而不是恰好复用了上一轮算对的数据——上一轮 122 的 256³「通过」就是这么假过的。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/123_mm_registration.sh --name 123_mm_registration --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
# 走 C++ aten::mm 需要这个开关（默认关：Lead 先保正确性）。
export COREX_FAST_MM=1
echo "HOST=$(hostname) DATE=$(date -Is) COREX_FAST_MM=${COREX_FAST_MM}"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/mm_shapes.py" <<'PYEOF'
"""形状矩阵 + 污染池 + 两条路径逐一与 CPU 对账。"""
import sys

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _ops_matmul, _runtime as rt

corex_torch.register()
shim = corex_torch._device.shim()
copy_from = corex_torch._device._copy_from_impl
DT = corex_torch._device._scalar_type(torch.float32)

print("shim 导出数:", len([n for n in dir(shim) if not n.startswith("_")]))
print("register() 后 mm 内核 dtype:", shim.mm_kernel_dtypes())
print("register() 后 mm 内核明细 [dtype, bm, bn, bk, warps, shared]:", shim.mm_kernel_info())

#: 形状矩阵：方阵 / 非方阵 / 极窄 / 非整除。
SHAPES = [
    (64, 256, 128, "64x128 @ 128x256"),
    (16, 64, 16, "16x16 @ 16x64"),
    (256, 256, 256, "256x256x256"),
    (128, 32, 64, "128x64 @ 64x32"),
    (37, 29, 53, "37x53 @ 53x29（非整除）"),
]
POISON = -12345.0


def poison_pool(M, N):
    """先占一块同尺寸的池块并填哨兵，再释放——让下一次同尺寸分配拿到脏块。"""
    src = torch.full((M, N), POISON, dtype=torch.float32)
    src_corex = src.to("corex")
    tmp = shim.empty_strided([M, N], [N, 1], DT, 0)
    copy_from(src_corex, tmp)                      # 把哨兵写进池块
    del tmp, src_corex
    import gc

    gc.collect()
    rt.synchronize()


def check_shape(M, N, K, label, mode):
    """跑一次并逐元素对 CPU；返回 (max_abs_diff, max_rel, 有没有残留哨兵)。"""
    torch.manual_seed(1234)
    a_cpu = torch.randn(M, K) * 0.5
    b_cpu = torch.randn(K, N) * 0.5
    poison_pool(M, N)
    a = a_cpu.to("corex")
    b = b_cpu.to("corex")
    if mode == "aten":
        out = torch.mm(a, b)                       # 走 C++ aten::mm
    else:
        out = shim.empty_strided([M, N], [N, 1], DT, 0)
        # spec 末尾的 p 是 global scratch（Triton 3.4 把它放在参数表最后）
        h = shim.load_kernel(_KERNEL.asm["cubin"], _KERNEL.name, "pppiiip",
                             _SHARED)
        shim.launch(h, [(M + 15) // 16, (N + 15) // 16, 1], [32, 1, 1], _SHARED, 0,
                    [a.data_ptr(), b.data_ptr(), out.data_ptr(), 0],
                    [], [M, N, K], [])
    rt.synchronize()
    back = torch.empty(M, N, dtype=torch.float32)
    copy_from(out, back)
    want = a_cpu @ b_cpu
    diff = (back - want).abs()
    max_abs = diff.max().item()
    max_rel = (diff / want.abs().clamp(min=1.0)).max().item()
    poisoned = int((back == POISON).sum().item())
    flag = "OK  " if max_abs <= 1e-4 * max(1.0, want.abs().max().item()) else "FAIL"
    print(f"  {flag} {label:22s} {mode:6s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"残留哨兵={poisoned}")
    return max_abs, max_rel, poisoned


# `_aten_kernels` 目前用 spec "ppppiii"（scratch 放在第 4 位，是错的）——先由探针用正确 spec
# 覆盖登记，把 C++ 实现本身验对/验错；它那边的 spec + shared 要 Lead 改（见收尾报告）。
_KERNEL = _ops_matmul._get_kernel("fp32", 16, 16, 16, False, 1, 1)
_SHARED = int(getattr(_KERNEL.metadata, "shared", 0) or 0)
print(f"探针自编内核 {_KERNEL.name}  metadata.shared = {_SHARED}")
_h = shim.load_kernel(_KERNEL.asm["cubin"], _KERNEL.name, "pppiiip", _SHARED)
shim.set_mm_kernel(6, _h, 16, 16, 16, 1, _SHARED)
print("覆盖为探针登记后明细:", shim.mm_kernel_info())

print("\n=== 1. aten 路径（探针用正确 spec/shared 覆盖登记后）===")
results_aten = [check_shape(M, N, K, label, "aten") for M, N, K, label in SHAPES]

print("\n=== 2. shim.launch 直调路径（探针自己 load_kernel，带上 metadata.shared）===")
print(f"  复用上面那份内核（shared={_SHARED}）")
results_direct = [check_shape(M, N, K, label, "direct") for M, N, K, label in SHAPES]

print("\n=== 3. 用探针的 handle 覆盖登记，再走一次 aten 路径 ===")
handle = shim.load_kernel(_KERNEL.asm["cubin"], _KERNEL.name, "pppiiip", _SHARED)
shim.set_mm_kernel(6, handle, 16, 16, 16, 1)
print("  覆盖后明细:", shim.mm_kernel_info())
results_after = [check_shape(M, N, K, label, "aten") for M, N, K, label in SHAPES]

print("\n=== 4. 结论 ===")
aten_ok = all(abs_ <= 1e-4 for abs_, _, _ in results_aten)
direct_ok = all(abs_ <= 1e-4 for abs_, _, _ in results_direct)
after_ok = all(abs_ <= 1e-4 for abs_, _, _ in results_after)
print(f"  aten(现状) all_ok={aten_ok} | 直调 all_ok={direct_ok} | aten(探针 handle) all_ok={after_ok}")
if after_ok and not aten_ok:
    print("REGISTRATION_MISMATCH：错在登记给 C++ 的 (block/warps/shared/spec)，不在 launch 本身")
elif direct_ok and not after_ok:
    print("LAUNCH_OK_BUT_ATEN_PIECE_BAD")
elif direct_ok and after_ok:
    print("ALL_OK")
else:
    print("LAUNCH_OR_KERNEL_BAD：直调也错，问题在 C++ launch 或内核参数展开")
print("MM_SHAPES_DONE")
raise SystemExit(0 if (after_ok and direct_ok) else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/mm_shapes.py" "$W/src"
RC=$?
echo "MM_SHAPES_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE123_DONE $(date -Is)"
exit $RC
