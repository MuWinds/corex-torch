set -u
# 探针 122：C++ 快速启动路径 vs Triton 的 Python 启动路径（真机分段计时）。
#
# 为什么做这个：probes/010 的归因表里，每次算子的纯 host 开销 ≈100 µs（启动 39.6 + aten 59.7），
# 与 mm 256³ 的内核时间（≈166 µs）同量级。coffin 的 decode 每 token 六百多次发射，
# 按这个数就是每步 60 ms 级白费。本探针量两件事：
#   1. 把「实参打包 + 多层 Python」换成一次 `cuLaunchKernel` 之后，单次发射降到多少；
#   2. 同一份 cubin 走两条路径（Triton Python / shim 快路径）的数值是否一致。
#
# 口径：单卡（runner 的 CVD）、fp32、mm 256³ 与 16³、循环外同步一次（与 probes/010 对齐）。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/122_fast_launch.sh --name 122_fast_launch --tail 0

PY=$HOME/venvs/corex28/bin/python
W=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}
export COREX_TRITON_SRC=$W/corex-triton-src
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > "$W/fast_launch.py" <<'PYEOF'
"""量两条启动路径，并校验数值一致。"""
import sys
import time

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

import corex_torch
from corex_torch import _ops_matmul, _runtime as rt, _triton

corex_torch.register()
shim = corex_torch._device.shim()
scalar_type = corex_torch._device._scalar_type
DT = scalar_type(torch.float32)
print("shim 导出数:", len([n for n in dir(shim) if not n.startswith("_")]))
# 先看 `_aten_kernels.register()` 有没有在**注册期**就把 mm 内核登记进来（Lead 的链路）。
# 期望至少看到 fp32=6；fp16=5 要等内核组把 fp16/b f16 内核接上。
print("register() 之后已登记 mm 内核的 dtype:", shim.mm_kernel_dtypes())

# --- 编译一份 mm 内核（与 _ops_matmul 用的完全同一份），取 cubin 与内核名 ---
kernel = _ops_matmul._get_kernel("fp32", 16, 16, 16, False, 1, 1)
cubin = kernel.asm["cubin"]
name = getattr(kernel, "name", None) or kernel.metadata.name
print(f"内核 {name}  cubin {len(cubin)} B")
# 实参类型串：a/b/c 三个设备指针 + **global scratch 指针**（Triton 3.4 在末尾追加，
# 少传这一个会让内核从栈上读到垃圾指针 → 段错误，2026-11-07 实测）+ M/N/K 三个 i32。
scratch_bytes = int(getattr(kernel.metadata, "global_scratch_size", 0) or 0)
print(f"global_scratch_size = {scratch_bytes}")
SPEC = "ppppiii"
handle = shim.load_kernel(cubin, name, SPEC)
handle2 = shim.load_kernel(cubin, name, SPEC)
print(f"load_kernel 句柄={handle} 二次装载复用同一句柄={handle == handle2} "
      f"已缓存内核数={shim.kernel_count()}")
# 登记给 C++ 的 aten::mm（fp32 = c10::ScalarType 6）。若 `_aten_kernels.register()` 已经登记过，
# 这里就**不重复登记**——重复会把「Lead 的链路是否真的通」这件事盖住。
if 6 in shim.mm_kernel_dtypes():
    print("fp32 已由 _aten_kernels 在注册期登记，探针不重复登记")
else:
    shim.set_mm_kernel(6, handle, 16, 16, 16, 1)
    print("fp32 **未**被注册期登记，探针自行补登记（说明 Lead 的链路未生效）")
print("最终已登记 mm 内核的 dtype:", shim.mm_kernel_dtypes())


def bench(label: str, fn, iters: int = 200):
    """循环外同步一次，返回单次平均微秒（与 probes/010 同口径）。"""
    fn()
    rt.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    rt.synchronize()
    per_iter_us = (time.perf_counter() - started) / iters * 1e6
    print(f"  {label:36s} {per_iter_us:8.2f} us/iter")
    return per_iter_us


def make(n: int, seed: float):
    """造一个 n×n 的 corex 张量（CPU 造好后搬过去）。"""
    cpu = torch.arange(n * n, dtype=torch.float32).reshape(n, n) * seed
    return cpu.to("corex")


print("\n=== 1. mm 16³：把内核时间榨到近 0，看纯发射地板 ===")
a16, b16, c16 = make(16, 0.01), make(16, 0.02), torch.empty(16, 16, device="corex")
grid16 = [(16 + 15) // 16, (16 + 15) // 16, 1]
block = [32, 1, 1]
t_py16 = bench("Triton Python 启动（_ops_matmul.mm）",
               lambda: _ops_matmul.mm(a16.data_ptr(), b16.data_ptr(), c16.data_ptr(),
                                      16, 16, 16, "fp32"))
t_cc16 = bench("shim.launch（C++ 快路径）",
               lambda: shim.launch(handle, grid16, block, 0, 0,
                                   [a16.data_ptr(), b16.data_ptr(), c16.data_ptr(), 0],
                                   [], [16, 16, 16], []))

print("\n=== 2. mm 256³：内核占主导，看 host 开销还剩多少 ===")
n = 256
a, b, c = make(n, 0.001), make(n, 0.001), torch.empty(n, n, device="corex")
grid = [(n + 15) // 16, (n + 15) // 16, 1]
flops = 2.0 * n * n * n
t_py = bench("Triton Python 启动（_ops_matmul.mm）",
             lambda: _ops_matmul.mm(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, n, n, "fp32"),
             iters=100)
t_cc = bench("shim.launch（C++ 快路径）",
             lambda: shim.launch(handle, grid, block, 0, 0,
                                 [a.data_ptr(), b.data_ptr(), c.data_ptr(), 0],
                                 [], [n, n, n], []),
             iters=100)
print(f"  GFLOP/s：Python 路径 {flops / (t_py * 1e-6) / 1e9:7.1f} | "
      f"C++ 快路径 {flops / (t_cc * 1e-6) / 1e9:7.1f}")

print("\n=== 2.5 aten 层：`a @ b` 现在由 C++ 的 aten::mm 接（D 段）===")
t_aten = bench("aten a @ b（C++ aten::mm）", lambda: a @ b, iters=100)
print(f"  GFLOP/s：aten 路径 {flops / (t_aten * 1e-6) / 1e9:7.1f}"
      f"（probes/010 基线是 265.8 us / 126.2 GFLOP/s）")
aten_out = a @ b
rt.synchronize()
back_aten = torch.empty(n, n, dtype=torch.float32)
corex_torch._device._copy_from_impl(aten_out, back_aten)
aten_ok = torch.equal(back_aten, a.cpu() @ b.cpu())
print(f"  aten 路径与 CPU 逐元素相等: {aten_ok}")

print("\n=== 2.6 把 aten 那多出来的 ~94 us 拆开 ===")
# 逐个量「各层各加了多少」，好定位到具体一层（不是猜）。
bench("torch.mm(a, b)                     ", lambda: torch.mm(a, b), iters=100)
bench("a @ b                               ", lambda: a @ b, iters=100)


def alloc_then_launch():
    """模拟 aten 路径：每轮新分配输出 + 直调快路径（不经分发器）。"""
    tmp = shim.empty_strided([n, n], [n, 1], DT, 0)
    shim.launch(handle, grid, block, 0, 0, [a.data_ptr(), b.data_ptr(), tmp.data_ptr(), 0],
                [], [n, n, n], [])
    return tmp


bench("分配 + 直调快路径（模拟 aten）       ", alloc_then_launch, iters=100)
# 纯 Python->C++ 调用地板：任何一次 pybind 调用都要付这笔钱。
bench("shim.is_initialized()（pybind 地板） ", lambda: shim.is_initialized(), iters=1000)
# 16³ 走 aten：内核几乎为 0，量到的就是「分发 + 分配 + 发射」的全部开销。
k16 = _ops_matmul._get_kernel("fp32", 16, 16, 16, False, 1, 1)
shim.set_mm_kernel(6, shim.load_kernel(k16.asm["cubin"], k16.name, SPEC), 16, 16, 16, 1)
bench("aten mm 16³（分发地板）             ", lambda: torch.mm(a16, b16), iters=200)
try:
    bench("torch.mm(a, b, out=c) 复用输出      ", lambda: torch.mm(a, b, out=c), iters=100)
except Exception as exc:                            # noqa: BLE001 - mm.out 未实现是已知缺口
    print(f"  torch.mm(a, b, out=c)              跳过：{type(exc).__name__}（aten::mm.out 未注册）")

print("\n=== 3. 数值一致性：两条路径算出来的 c 必须完全相等 ===")
# 先让快路径写一遍，读回；再让 Python 路径写一遍，读回；两者逐元素比。
back_cc = torch.empty(n, n, dtype=torch.float32)
corex_torch._device._copy_from_impl(c, back_cc)
shim.launch(handle, grid, block, 0, 0, [a.data_ptr(), b.data_ptr(), c.data_ptr(), 0],
            [], [n, n, n], [])
rt.synchronize()
back_cc2 = torch.empty(n, n, dtype=torch.float32)
corex_torch._device._copy_from_impl(c, back_cc2)
_ops_matmul.mm(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, n, n, "fp32")
rt.synchronize()
back_py = torch.empty(n, n, dtype=torch.float32)
corex_torch._device._copy_from_impl(c, back_py)
same = torch.equal(back_cc2, back_py)
print(f"  两条路径结果逐元素相等: {same}")
print(f"  与 CPU 参照的 max_abs_diff: {(back_py - (a.cpu() @ b.cpu())).abs().max().item():.3e}")

print("\n=== 4. 汇总（口径：单卡 fp32，循环外同步一次，200/100 次取平均）===")
print(f"  mm 16³  纯发射：Python {t_py16:7.2f} us -> C++ {t_cc16:7.2f} us "
      f"（{t_py16 / t_cc16:.1f}x）")
print(f"  mm 256³ 端到端：Python {t_py:7.2f} us -> C++ {t_cc:7.2f} us "
      f"（{t_py / t_cc:.2f}x）")
print(f"  mm 256³ aten（a @ b）：{t_aten:7.2f} us / "
      f"{flops / (t_aten * 1e-6) / 1e9:.1f} GFLOP/s"
      f"（probes/010 基线 265.8 us / 126.2 GFLOP/s）")
# 判据说明：aten 路径要跟**它自己的历史基线**比（265.8 us），不能跟「Python 直调 + 预分配输出」
# 比——后者少一次分配、也没有分发器，两者不是同一件事。
launch_ok = t_cc < t_py and t_cc16 < t_py16
aten_ok_speed = t_aten < 265.8
ok_all = same and aten_ok and launch_ok and aten_ok_speed
print(f"  判据：发射提速 {launch_ok} / aten 优于 010 基线 {aten_ok_speed} / 数值一致 "
      f"{same and aten_ok}")
print("FAST_LAUNCH_PASS" if ok_all else "FAST_LAUNCH_FAIL")
raise SystemExit(0 if ok_all else 1)
PYEOF

timeout --foreground -k 5 280 "$PY" -B -u "$W/fast_launch.py" "$W/src"
RC=$?
echo "FAST_LAUNCH_RC=$RC"

echo
echo "=== 收尾验卡 ==="
if timeout --foreground -k 2 20 /data/software/Anaconda3/2024.10-1/envs/py310/bin/python -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
echo "PROBE122_DONE $(date -Is)"
exit $RC
