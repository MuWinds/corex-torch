set -u
# 目的：验收元素级算子的第一批实现（`src/corex_torch/_ops_elementwise.py`），三件事一起量：
#   1. 正确性——与 CPU 参照**逐字节相等**（silu 除外，它与 `ex2.approx.f32` 有关，用最大相对误差）；
#   2. 尾块——n 取 BLOCK(1024) 的非整数倍：4097 与 4000 两组，验证 mask 没漏没越界；
#   3. 性能——n = 1<<20 fp32 下每个算子的 GB/s，与三个参照并列：
#        我们的 D2D memcpy（同一进程实测）、Triton 逐元素基线 295 GB/s、ATen elementwise 402 GB/s
#        （后两个来自 corex-triton/docs/00-FACTS.md §14.3 与 docs/00-FACTS.md §9，只当天花板）。
#
# 为什么这层必须能在「无 torch 进程」里验：算子层刻意不 import torch（接口是设备指针 + n +
# dtype 字符串），这样 C++ shim 就位之前就能在真机上跑出数。
#
# 为什么口径写成 buffers：不同算子的访存流量不同（二元读 2 写 1 = 3 份，一元读 1 写 1 = 2 份，
# fill 只写 1 份）。分母搞错会让「带宽」看起来凭空翻倍，所以流量由 `ops.traffic_bytes(n, dtype, k)`
# 显式算出并打印，每个算子带自己的 k。
#
# 载荷（本机先做一次，再跑本脚本）：
#   tar czf tb_ops.tgz -C <repo 根> corex-torch/src/corex_torch corex-triton/src/triton_corex
#   python tools/remote.py put tb_ops.tgz <home>/coffin-work/base/sandbox/work/ops/payload.tgz
#   python tools/remote.py sbx-run probes/170_eltwise.sh --name 170_eltwise --timeout 600
#
# 判定：末段 ELTWISE_VERDICT PASS，且退出码如实反映（脚本最后一句若只是 echo，runner 里永远 rc=0）。

PY=$HOME/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
mkdir -p "$W"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES，不要覆盖）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

echo "=== 1. 解包载荷 ==="
if [ ! -f "$W/payload.tgz" ]; then echo "PAYLOAD_MISSING $W/payload.tgz"; exit 66; fi
rm -rf "$W/corex-torch" "$W/corex-triton"
tar xzf "$W/payload.tgz" -C "$W" || exit 67
ls "$W/corex-torch/src/corex_torch/_ops_elementwise.py" "$W/corex-triton/src/triton_corex/driver.py" || exit 67
export PYTHONPATH=$W/corex-torch/src
export COREX_TRITON_SRC=$W/corex-triton/src

cat > "$W/bench_eltwise.py" <<'PYEOF'
"""元素级算子第一批：正确性（含尾块）+ 性能 + 参照。无 torch 进程。"""

import ctypes
import math
import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _triton as ct
from corex_torch import _ops_elementwise as ops


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


# --------------------------------------------------------------------------- #
# 主机侧打包 / 解包（fp32 / fp16 / bf16）
# --------------------------------------------------------------------------- #

FMT = {"fp32": "<f", "fp16": "<e"}


def f32_to_bf16_bits(v):
    """float32 → bfloat16 位模式，round-to-nearest-even（与硬件转换一致）。"""
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) >> 16
    return u & 0xFFFF


def round_to_dtype(v, dtype):
    """把一个 double 舍入到目标 dtype 再还原成 Python float（一次舍入，模拟硬件语义）。"""
    if dtype == "bf16":
        bits = f32_to_bf16_bits(struct.unpack("<f", struct.pack("<f", v))[0])
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    return struct.unpack(FMT[dtype], struct.pack(FMT[dtype], v))[0]


def pack(values, dtype):
    if dtype == "bf16":
        return b"".join(struct.pack("<H", f32_to_bf16_bits(v)) for v in values)
    return b"".join(struct.pack(FMT[dtype], v) for v in values)


def unpack(raw, n, dtype):
    if dtype == "bf16":
        out = []
        for i in range(n):
            bits = struct.unpack_from("<H", raw, 2 * i)[0]
            out.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
        return out
    fmt = FMT[dtype]
    size = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, raw, size * i)[0] for i in range(n)]


def to_dev(values, dtype):
    raw = pack(values, dtype)
    ptr = rt.malloc(len(raw))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr, len(raw)


def from_dev(ptr, n, dtype):
    size = ops.itemsize(dtype) * n
    buf = ctypes.create_string_buffer(size)
    rt.memcpy(buf, ptr, size, rt.MEMCPY_DEVICE_TO_HOST)
    return unpack(buf.raw, n, dtype)


# --------------------------------------------------------------------------- #
# 测试数据：全部取 fp32/fp16/bf16 都能精确表示的值
#    a ∈ [-12.0, 12.0] 步长 0.25；b 取 2 的幂（0.5/1.0/2.0/4.0）
#    b 取 2 的幂是关键：div 的商在 fp32 里再舍一次到 fp16 也不会出现双舍入差异，
#    这样 div 也能按「逐字节相等」判定，而不是躲在误差里。
# --------------------------------------------------------------------------- #

def make_data(n):
    a = [((i % 97) - 48) * 0.25 for i in range(n)]
    b = [[0.5, 1.0, 2.0, 4.0][i % 4] for i in range(n)]
    return a, b


def max_abs(got, ref):
    return max((abs(g - e) for g, e in zip(got, ref)), default=0.0)


def max_rel(got, ref):
    m = 0.0
    for g, e in zip(got, ref):
        d = abs(g - e)
        s = abs(e) if abs(e) > 1e-30 else 1.0
        m = max(m, d / s)
    return m


def first_bad(got_bytes_a, got_bytes_b):
    """逐字节比较，返回第一个不一致的下标（None 表示完全一致）。"""
    for i in range(min(len(got_bytes_a), len(got_bytes_b))):
        if got_bytes_a[i] != got_bytes_b[i]:
            return i
    return None if len(got_bytes_a) == len(got_bytes_b) else min(len(got_bytes_a), len(got_bytes_b))


# --------------------------------------------------------------------------- #
# 1. 正确性 + 尾块
# --------------------------------------------------------------------------- #

def correctness(dtype):
    """返回 (通过数, 总数, 明细行)。"""
    print(f"=== 正确性 dtype={dtype} ===")
    rows = []
    ok = 0
    total = 0
    # 4096 = BLOCK 整数倍；4097 / 4000 是尾块（分别只多 1 个、少 96 个）
    for n in (4096, 4097, 4000):
        a, b = make_data(n)
        pa, nbytes = to_dev(a, dtype)
        pb, _ = to_dev(b, dtype)
        po = rt.malloc(nbytes)
        pack_fmt = FMT.get(dtype)

        def ref_bin(fn):
            return [round_to_dtype(fn(x, y), dtype) for x, y in zip(a, b)]

        def check(name, call, expect, exact=True, tol=0.0):
            nonlocal ok, total
            call()
            rt.synchronize()
            got = from_dev(po, n, dtype)
            total += 1
            got_raw = pack(got, dtype)
            exp_raw = pack(expect, dtype)
            bad = first_bad(got_raw, exp_raw)
            if exact:
                good = bad is None
                detail = f"逐字节相等={good}" + ("" if good else f" 首个不同偏移={bad}")
            else:
                rel = max_rel(got, expect)
                good = rel <= tol
                detail = (f"max_abs={max_abs(got, expect):.3e} max_rel={rel:.3e} "
                          f"<= {tol:.1e} → {good}")
            ok += 1 if good else 0
            rows.append((f"{name:12s} n={n:5d}", detail))
            print(f"  {name:12s} n={n:5d}  {detail}")
            return good

        # 二元张量-张量
        check("add", lambda: ops.add(pa, pb, po, n, dtype), ref_bin(lambda x, y: x + y))
        check("add(alpha=.5)", lambda: ops.add(pa, pb, po, n, dtype, 0.5),
              ref_bin(lambda x, y: x + 0.5 * y))
        check("add(alpha=2.5)", lambda: ops.add(pa, pb, po, n, dtype, 2.5),
              ref_bin(lambda x, y: x + 2.5 * y))
        check("sub", lambda: ops.sub(pa, pb, po, n, dtype), ref_bin(lambda x, y: x - y))
        check("sub(alpha=2)", lambda: ops.sub(pa, pb, po, n, dtype, 2.0),
              ref_bin(lambda x, y: x - 2.0 * y))
        check("mul", lambda: ops.mul(pa, pb, po, n, dtype), ref_bin(lambda x, y: x * y))
        check("div", lambda: ops.div(pa, pb, po, n, dtype), ref_bin(lambda x, y: x / y))

        # 二元张量-标量
        check("add_scalar", lambda: ops.add_scalar(pa, po, n, dtype, 0.5),
              ref_bin(lambda x, _y: x + 0.5))
        check("sub_scalar", lambda: ops.sub_scalar(pa, po, n, dtype, 2.0),
              ref_bin(lambda x, _y: x - 2.0))
        check("mul_scalar", lambda: ops.mul_scalar(pa, po, n, dtype, 0.25),
              ref_bin(lambda x, _y: x * 0.25))
        check("div_scalar", lambda: ops.div_scalar(pa, po, n, dtype, 4.0),
              ref_bin(lambda x, _y: x / 4.0))

        # 一元
        check("fill(0.5)", lambda: ops.fill(po, n, dtype, 0.5),
              [round_to_dtype(0.5, dtype)] * n)
        check("fill(-2.0)", lambda: ops.fill(po, n, dtype, -2.0),
              [round_to_dtype(-2.0, dtype)] * n)
        check("copy", lambda: ops.copy(pa, po, n, dtype),
              [round_to_dtype(x, dtype) for x in a])
        # silu：唯一不与 CPU 逐位相等的算子（ex2.approx.f32），单独用相对误差
        check("silu", lambda: ops.silu(pa, po, n, dtype),
              [round_to_dtype(x / (1.0 + math.exp(-x)), dtype) for x in a],
              exact=False, tol=1e-6)

        rt.free(pa)
        rt.free(pb)
        rt.free(po)
    return ok, total


# --------------------------------------------------------------------------- #
# 2. 内核缓存行为：同一配置第二次调用不应该再编译
# --------------------------------------------------------------------------- #

def cache_check():
    before = ops.cache_info()["kernels"]
    n = 4096
    raw = 4096 * 4
    pa = rt.malloc(raw)
    pb = rt.malloc(raw)
    po = rt.malloc(raw)
    t0 = time.perf_counter()
    ops.add(pa, pb, po, n, "fp32")           # 已编过 → 应命中缓存
    rt.synchronize()
    dt = 1000 * (time.perf_counter() - t0)
    after = ops.cache_info()["kernels"]
    print(f"=== 缓存：已编内核 {before} → {after}，一次 add 调用 {dt:.3f} ms")
    for p in (pa, pb, po):
        rt.free(p)
    return after == before


# --------------------------------------------------------------------------- #
# 3. 性能：n = 1<<20 fp32
# --------------------------------------------------------------------------- #

def bench(fn, reps=20, warmup=3):
    """返回每次调用的秒数。**含 Python 派发开销**——这是 `_device.py` 实际付的账。"""
    for _ in range(warmup):
        fn()
    rt.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    rt.synchronize()
    return (time.perf_counter() - t0) / reps


#: L2 是 32 MiB（`probes/150` 属性 38）。刷 L2 的缓冲取 256 MiB，与 do_bench 的 256 MB 同口径：
#: 不刷的话 3×4 MiB 的工作集全在 L2 里，量出来的是 L2 带宽而不是显存带宽
#: （第一次跑就吃了这个亏：add 报 594 GB/s，比 D2D memcpy 还高）。
_FLUSH = {}


def flush_l2():
    """用一次 256 MiB 的 fill 把 L2 顶掉（写流量 256 MiB ≫ 32 MiB L2）。"""
    if "ptr" not in _FLUSH:
        n = 64 << 20                                   # 64M 个 fp32 = 256 MiB
        _FLUSH["ptr"] = rt.malloc(n * 4)
        _FLUSH["n"] = n
    ops.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench_events(fn, reps=20, warmup=3, flush=True):
    """按 do_bench 的口径计时：每轮只把 `fn` 包在两个事件之间，flush 与派发都不进计时窗口。

    Returns:
        (中位数秒, 最小秒)。中位数抗抖动，最小值代表上限。
    """
    pairs = [(rt.Event(), rt.Event()) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    rt.synchronize()
    for i in range(reps):
        if flush:
            flush_l2()
        pairs[i][0].record()
        fn()
        pairs[i][1].record()
    rt.synchronize()
    ts = sorted(a.elapsed_ms(b) / 1000.0 for a, b in pairs)
    for a, b in pairs:
        a.close()
        b.close()
    return ts[len(ts) // 2], ts[0]


def _lib_launch(kernel, params, grid_x, block_x):
    """预打包参数后直接调 cuLaunchKernel 的闭包：排除 Python 派发开销。"""
    lib = ct._LibCuda.get()
    fn = kernel.function
    shared = int(kernel.metadata.shared)

    def go():
        lib.cuLaunchKernel(fn, grid_x, 1, 1, block_x, 1, 1, shared, None, params, None)

    return go


def _params(objs):
    return (ctypes.c_void_p * len(objs))(
        *[ctypes.cast(ctypes.byref(o), ctypes.c_void_p) for o in objs])


def _kernel(kind, dtype, op, block, num_warps=None):
    k = ops._get_kernel(kind, dtype, op, block, num_warps or ops.DEFAULT_NUM_WARPS)
    k._init_handles()                       # 触发装载，之后 k.function 可用
    return k


def performance():
    n = 1 << 20
    dtype = "fp32"
    a, b = make_data(n)
    pa, nbytes = to_dev(a, dtype)
    pb, _ = to_dev(b, dtype)
    po = rt.malloc(nbytes)
    paddr, pbaddr, poaddr = pa.value, pb.value, po.value
    scratch = ctypes.c_void_p(0)
    block = ops.DEFAULT_BLOCK
    gw = ops.DEFAULT_NUM_WARPS * 32
    grid_x = (n + block - 1) // block
    kib = 1024.0

    # --- 3.1 内核本身：预打包参数、L2 已刷（与 295/402 两个参照同口径）---
    layout = {
        "add":        ("binary", 0, 3, [ctypes.c_void_p(paddr), ctypes.c_void_p(pbaddr),
                                        ctypes.c_void_p(poaddr), ctypes.c_int32(n),
                                        ctypes.c_float(1.0), scratch], 3),
        "sub":        ("binary", 1, 3, [ctypes.c_void_p(paddr), ctypes.c_void_p(pbaddr),
                                        ctypes.c_void_p(poaddr), ctypes.c_int32(n),
                                        ctypes.c_float(1.0), scratch], 3),
        "mul":        ("binary", 2, 3, [ctypes.c_void_p(paddr), ctypes.c_void_p(pbaddr),
                                        ctypes.c_void_p(poaddr), ctypes.c_int32(n),
                                        ctypes.c_float(1.0), scratch], 3),
        "div":        ("binary", 3, 3, [ctypes.c_void_p(paddr), ctypes.c_void_p(pbaddr),
                                        ctypes.c_void_p(poaddr), ctypes.c_int32(n),
                                        ctypes.c_float(1.0), scratch], 3),
        "add_scalar": ("scalar", 0, 2, [ctypes.c_void_p(paddr), ctypes.c_void_p(poaddr),
                                        ctypes.c_int32(n), ctypes.c_float(0.5), scratch], 2),
        "fill":       ("fill", None, 1, [ctypes.c_void_p(poaddr), ctypes.c_int32(n),
                                         ctypes.c_float(0.5), scratch], 1),
        "silu":       ("silu", None, 2, [ctypes.c_void_p(paddr), ctypes.c_void_p(poaddr),
                                         ctypes.c_int32(n), scratch], 2),
        "copy":       ("copy", None, 2, [ctypes.c_void_p(paddr), ctypes.c_void_p(poaddr),
                                         ctypes.c_int32(n), scratch], 2),
    }
    print(f"=== 3.1 内核本身（预打包 + 直调 cuLaunchKernel，L2 每轮刷 256 MiB）"
          f" n={n} fp32 ===")
    print(f"  {'算子':12s} {'流量':>9s} {'耗时中位':>10s} {'GB/s':>8s} {'最小耗时':>10s}")
    kern = {}
    for name, (kind, op, _dummy, objs, buffers) in layout.items():
        k = _kernel(kind, dtype, op, block)
        go = _lib_launch(k, _params(objs), grid_x, gw)
        med, best = bench_events(go, reps=20)
        tb = ops.traffic_bytes(n, dtype, buffers)
        kern[name] = (med, tb / med / 1e9)
        print(f"  {name:12s} {tb / 1048576:8.2f} MiB {med * 1e6:8.1f} us "
              f"{tb / med / 1e9:8.1f} {best * 1e6:8.1f} us")

    # --- 3.2 端到端（走 ops 层）+ 派发开销 + L2 热的口径 ---
    print(f"=== 3.2 ops 层端到端（墙钟，含 Python 派发）与 L2 热口径 ===")
    print(f"  {'算子':12s} {'墙钟/次':>10s} {'派发':>8s} {'GB/s(含派发)':>13s} "
          f"{'L2热GB/s':>10s}")
    e2e = {}
    calls = {
        "add":        (lambda: ops.add(pa, pb, po, n, dtype), 3),
        "sub":        (lambda: ops.sub(pa, pb, po, n, dtype), 3),
        "mul":        (lambda: ops.mul(pa, pb, po, n, dtype), 3),
        "div":        (lambda: ops.div(pa, pb, po, n, dtype), 3),
        "add_scalar": (lambda: ops.add_scalar(pa, po, n, dtype, 0.5), 2),
        "fill":       (lambda: ops.fill(po, n, dtype, 0.5), 1),
        "silu":       (lambda: ops.silu(pa, po, n, dtype), 2),
        "copy":       (lambda: ops.copy(pa, po, n, dtype), 2),
    }
    for name, (fn, buffers) in calls.items():
        wall = bench(fn, reps=20)
        hot, _ = bench_events(fn, reps=20, flush=False)      # L2 热（不刷）
        tb = ops.traffic_bytes(n, dtype, buffers)
        disp = (wall - hot) * 1e6                            # 派发 = 墙钟 − L2 热的内核时间
        e2e[name] = (wall, tb / wall / 1e9)
        print(f"  {name:12s} {wall * 1e6:8.1f} us {disp:6.1f} us {tb / wall / 1e9:10.1f} "
              f"{tb / hot / 1e9:10.1f}")

    # div 在第一版探针里出现过一次 581 us 的离群值（同配置复跑不再出现），这里再量一次留证。
    k = _kernel("binary", dtype, 3, block)
    go = _lib_launch(k, _params(layout["div"][3]), grid_x, gw)
    med2, _ = bench_events(go, reps=20)
    print(f"  {'div 复测':12s} {med2 * 1e6:8.1f} us（判离群用）"
          f" {ops.traffic_bytes(n, dtype, 3) / med2 / 1e9:8.1f} GB/s")

    # --- 3.3 同尺寸 DMA 拷贝 ---
    med, _ = bench_events(lambda: rt.memcpy(po, pa, nbytes, rt.MEMCPY_DEVICE_TO_DEVICE),
                          reps=20, flush=False)
    tb = ops.traffic_bytes(n, dtype, 2)
    print(f"  {'memcpy D2D':12s} {med * 1e6:8.1f} us {'—':>8s} {tb / med / 1e9:10.1f}  "
          f"← 连续 copy 的正确通道（同一块 8 MiB，L2 热）")
    kern["memcpy_d2d"] = (med, tb / med / 1e9)

    # --- 3.4 BLOCK / num_warps 扫描（内核本身，L2 刷）---
    print("=== 3.4 BLOCK/num_warps 扫描（add，内核本身口径，L2 刷）===")
    for blk, warps in ((128, 4), (256, 4), (512, 4), (1024, 4), (2048, 4), (4096, 8)):
        k = _kernel("binary", dtype, 0, blk, warps)
        objs = [ctypes.c_void_p(paddr), ctypes.c_void_p(pbaddr), ctypes.c_void_p(poaddr),
                ctypes.c_int32(n), ctypes.c_float(1.0), scratch]
        go = _lib_launch(k, _params(objs), (n + blk - 1) // blk, warps * 32)
        med, _ = bench_events(go, reps=20)
        tb = ops.traffic_bytes(n, dtype, 3)
        print(f"  BLOCK={blk:5d} warps={warps}  {med * 1e6:8.1f} us  {tb / med / 1e9:8.1f} GB/s")

    for p in (pa, pb, po):
        rt.free(p)
    print("=== 3.5 参照（同一口径：1M fp32、3 份流量、L2 冷）===")
    print("  Triton 逐元素基线（corex-triton §14.3，do_bench 也刷 L2）  295.0 GB/s")
    print("  ATen elementwise（docs/00-FACTS §9；本探针下一段同尺寸实测）402.0 / 377.9 GB/s")
    print(f"  我们 add 内核本身                                          {kern['add'][1]:.1f} GB/s"
          f"  → 相对 295 基线 {kern['add'][1] / 295.0:.2f}x，相对 ATen {kern['add'][1] / 377.9:.2f}x")
    print(f"  我们 add 端到端（含派发）                                  {e2e['add'][1]:.1f} GB/s")
    return kern, e2e


def main():
    print("[start] torch_modules =", len(torchmods()), torchmods()[:3])
    print("device =", rt.device_name(0),
          " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE),
          " runtime_version =", rt.runtime_version())
    print("DEFAULT_BLOCK =", ops.DEFAULT_BLOCK, " DEFAULT_NUM_WARPS =", ops.DEFAULT_NUM_WARPS)

    ok, total = correctness("fp32")
    cache_ok = cache_check()
    perf = performance()
    print("[end] torch_modules =", len(torchmods()), torchmods()[:3])

    verdict = "PASS" if (ok == total and cache_ok) else "FAIL"
    print(f"正确性 {ok}/{total}  缓存未重编={cache_ok}")
    print("ELTWISE_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 算子正确性 + 尾块 + 性能（无 torch 进程）==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/bench_eltwise.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 同一尺寸下 ATen elementwise 的成绩（厂商 torch 2.1，只当天花板）==="
timeout --foreground -k 5 120 "$PY_VENDOR" -B -u - <<'PY2'
import time
import torch
n = 1 << 20
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")
c = torch.empty(n, device="cuda")
for _ in range(3):
    torch.add(a, b, out=c)
torch.cuda.synchronize()
reps = 20
t0 = time.perf_counter()
for _ in range(reps):
    torch.add(a, b, out=c)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / reps
print("ATEN_SAME_N n=%d %.1f us %.1f GB/s" % (n, dt * 1e3, 3 * n * 4 / dt / 1e9))
PY2
ATENRC=$?
echo "ATEN_RC=$ATENRC"

echo "=== 4. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE170_DONE py_rc=$PYRC aten_rc=$ATENRC"
exit "$PYRC"
