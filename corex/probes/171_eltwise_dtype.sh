set -u
# 目的：把元素级算子第一批的**类型覆盖**量清楚，回答三个问题：
#   1. fp16 全算子是否与 CPU 参照逐字节相等（含尾块 4097 / 4000）；
#      特别覆盖 `fill(fp16, 0.5)`——`_device.py` 只在 value==0 时走 cudaMemset，
#      非 0 的 fp16 填充全靠这条内核，写错位宽就是静默垃圾；
#   2. bf16 在这条 llc 链路上到底能不能编（编不过必须显式 NotImplementedError，
#      绝不能静默降精度）；
#   3. fp16 下的带宽（n=1<<20；每个元素 2 字节，所以同样的 GB/s 口径下数值应与 fp32 接近）。
#
# 载荷与 170 相同（`work/ops/payload.tgz`）。
# 跑法：python tools/remote.py sbx-run probes/171_eltwise_dtype.sh --name 171_eltwise_dtype --timeout 600
# 判定：末段 DTYPE_VERDICT PASS，且 fp16 全通；bf16 的结论以 B 段打印为准（PASS/明确不支持都算信息）。

PY=$HOME/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
mkdir -p "$W"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES）==="
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
export PYTHONPATH=$W/corex-torch/src
export COREX_TRITON_SRC=$W/corex-triton/src

cat > "$W/dtype_matrix.py" <<'PYEOF'
"""类型矩阵：fp16 全通 + bf16 能不能编。无 torch 进程。"""

import ctypes
import math
import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as ops


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


FMT = {"fp32": "<f", "fp16": "<e"}


def f32_to_bf16_bits(v):
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def round_to_dtype(v, dtype):
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
    fmt, size = FMT[dtype], struct.calcsize(FMT[dtype])
    return [struct.unpack_from(fmt, raw, size * i)[0] for i in range(n)]


def to_dev(values, dtype):
    raw = pack(values, dtype)
    ptr = rt.malloc(len(raw))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr, len(raw)


def from_dev_raw(ptr, n, dtype):
    size = ops.itemsize(dtype) * n
    buf = ctypes.create_string_buffer(size)
    rt.memcpy(buf, ptr, size, rt.MEMCPY_DEVICE_TO_HOST)
    return buf.raw


def make_data(n):
    a = [((i % 97) - 48) * 0.25 for i in range(n)]
    b = [[0.5, 1.0, 2.0, 4.0][i % 4] for i in range(n)]
    return a, b


def max_rel(got, ref):
    m = 0.0
    for g, e in zip(got, ref):
        s = abs(e) if abs(e) > 1e-30 else 1.0
        m = max(m, abs(g - e) / s)
    return m


def run_dtype(dtype):
    """一个 dtype 的全算子正确性。返回 (通过, 总数, 不支持列表)。"""
    print(f"=== A. dtype={dtype} 正确性（含尾块）===")
    ok = total = 0
    unsupported = []
    for n in (4096, 4097, 4000):
        a, b = make_data(n)
        pa, nbytes = to_dev(a, dtype)
        pb, _ = to_dev(b, dtype)
        po = rt.malloc(nbytes)

        def check(name, call, expect, exact=True, tol=0.0):
            nonlocal ok, total
            total += 1
            try:
                call()
                rt.synchronize()
            except NotImplementedError as exc:
                unsupported.append(name)
                print(f"  {name:12s} n={n:5d}  NotImplementedError: {str(exc)[:110]}…")
                return False
            got_raw = from_dev_raw(po, n, dtype)
            exp_raw = pack(expect, dtype)
            if exact:
                good = got_raw == exp_raw
                detail = "逐字节相等=True" if good else "逐字节相等=False"
                if not good:
                    got = unpack(got_raw, n, dtype)
                    bad = next((i for i in range(n) if got_raw[2 * i:2 * i + 2] !=
                                exp_raw[2 * i:2 * i + 2]), -1)
                    detail += f" 首个不同下标={bad} got={got[bad] if bad >= 0 else '?'}"
            else:
                got = unpack(got_raw, n, dtype)
                rel = max_rel(got, expect)
                good = rel <= tol
                detail = f"max_rel={rel:.3e} <= {tol:.1e} → {good}"
            ok += 1 if good else 0
            print(f"  {name:12s} n={n:5d}  {detail}")

        check("add", lambda: ops.add(pa, pb, po, n, dtype), [a[i] + b[i] for i in range(n)])
        check("add(alpha=.5)", lambda: ops.add(pa, pb, po, n, dtype, 0.5),
              [a[i] + 0.5 * b[i] for i in range(n)])
        check("sub", lambda: ops.sub(pa, pb, po, n, dtype), [a[i] - b[i] for i in range(n)])
        check("mul", lambda: ops.mul(pa, pb, po, n, dtype), [a[i] * b[i] for i in range(n)])
        check("div", lambda: ops.div(pa, pb, po, n, dtype), [a[i] / b[i] for i in range(n)])
        check("add_scalar", lambda: ops.add_scalar(pa, po, n, dtype, 0.5),
              [a[i] + 0.5 for i in range(n)])
        check("mul_scalar", lambda: ops.mul_scalar(pa, po, n, dtype, 0.25),
              [a[i] * 0.25 for i in range(n)])
        check("fill(0.5)", lambda: ops.fill(po, n, dtype, 0.5),
              [round_to_dtype(0.5, dtype)] * n)
        check("fill(-2.0)", lambda: ops.fill(po, n, dtype, -2.0),
              [round_to_dtype(-2.0, dtype)] * n)
        check("copy", lambda: ops.copy(pa, po, n, dtype), [round_to_dtype(v, dtype) for v in a])
        check("silu", lambda: ops.silu(pa, po, n, dtype),
              [round_to_dtype(v / (1.0 + math.exp(-v)), dtype) for v in a],
              exact=False, tol=1e-6)

        # fill 的位模式证据：fp16 的 0.5 是 0x3800（小端 → 字节 00 38），bf16 是 0x3F00（00 3f）
        try:
            ops.fill(po, n, dtype, 0.5)
            rt.synchronize()
            head = from_dev_raw(po, 4, dtype)
            print(f"  fill(0.5) 前 4 个元素的字节 = {head.hex()}  "
                  f"（fp16 期望 0038003800380038 / bf16 期望 003f003f003f003f）")
        except NotImplementedError as exc:
            print(f"  fill(0.5) 位模式证据跳过：{str(exc)[:80]}…")

        rt.free(pa)
        rt.free(pb)
        rt.free(po)
    return ok, total, unsupported


_FLUSH = {}


def flush_l2():
    """刷 L2（32 MiB）：3 个 2 MiB 的 fp16 缓冲全在 L2 里，不刷就量成 L2 带宽。"""
    if "ptr" not in _FLUSH:
        m = 64 << 20
        _FLUSH["ptr"] = rt.malloc(m * 4)
        _FLUSH["n"] = m
    ops.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench_events(fn, reps=20, warmup=3, flush=True):
    """do_bench 口径：每轮只把 fn 包在两个事件之间（中位数）。"""
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
    return ts[len(ts) // 2]


def bench_fp16():
    n = 1 << 20
    dtype = "fp16"
    a = [((i % 97) - 48) * 0.25 for i in range(n)]
    b = [[0.5, 1.0, 2.0, 4.0][i % 4] for i in range(n)]
    pa, _ = to_dev(a, dtype)
    pb, _ = to_dev(b, dtype)
    po = rt.malloc(n * 2)
    print(f"=== C. fp16 带宽 n={n}（每缓冲 {n * 2 / 1048576:.1f} MiB，事件计时 + L2 刷）===")
    cases = (
        ("add", lambda: ops.add(pa, pb, po, n, dtype), 3),
        ("fill", lambda: ops.fill(po, n, dtype, 0.5), 1),
        ("silu", lambda: ops.silu(pa, po, n, dtype), 2),
        ("copy", lambda: ops.copy(pa, po, n, dtype), 2),
    )
    for name, fn, buffers in cases:
        med = bench_events(fn, reps=20)
        tb = ops.traffic_bytes(n, dtype, buffers)
        print(f"  {name:6s} {med * 1e6:8.1f} us  {tb / med / 1e9:8.1f} GB/s")
    # 复测一次 add：上一版墙钟口径在 fp16 add 上报过 619 us（10 GB/s）的离群值，
    # 与 copy 的 41 us 差 15 倍，明显不是「半精度的真实代价」，必须复查。
    med = bench_events(cases[0][1], reps=20)
    print(f"  add 复测 {med * 1e6:6.1f} us  "
          f"{ops.traffic_bytes(n, dtype, 3) / med / 1e9:8.1f} GB/s（判离群用）")
    med = bench_events(lambda: rt.memcpy(po, pa, n * 2, rt.MEMCPY_DEVICE_TO_DEVICE),
                       reps=20, flush=False)
    print(f"  memcpy {med * 1e6:6.1f} us  "
          f"{ops.traffic_bytes(n, dtype, 2) / med / 1e9:8.1f} GB/s（同一块 4 MiB，L2 热）")
    for p in (pa, pb, po):
        rt.free(p)


# --------------------------------------------------------------------------- #
# B. 16 位能不能绕过去：掩码模板缺失 vs 无掩码
# --------------------------------------------------------------------------- #

import triton
import triton.language as tl
from corex_torch import _triton as ct


@triton.jit
def _add_nomask(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """无掩码版本：只在 n 是 BLOCK 整数倍时合法。

    存在的意义：16 位（fp16/bf16）的**带掩码**访存模板在 corex-triton 的 `rewrite.py`
    里没有登记（`@$2 ld.global.b16`）。这条探针就是问「去掉掩码是不是就绕过去了」——
    如果是，那 fp16 在「n 对齐」的常见情形下现在就能用，不必等改写表扩充。
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, tl.load(a_ptr + off) + tl.load(b_ptr + off))


def probe_unmasked(dtype):
    """返回 True 表示该 dtype 的无掩码内核编得过且算得对。"""
    ptr_ty = ops.dtype_info(dtype)[0]
    try:
        k = ct.compile_kernel(_add_nomask,
                              {"a_ptr": ptr_ty, "b_ptr": ptr_ty, "out_ptr": ptr_ty,
                               "n": "i32", "BLOCK": "constexpr"},
                              {"BLOCK": 1024})
    except Exception as exc:                            # noqa: BLE001 - 失败也是结论
        print(f"  unmasked {dtype} 编译失败 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:150]}")
        return False
    print(f"  unmasked {dtype} 编译成功 cubin={len(k.asm['cubin'])} B")
    n = 4096
    a, b = make_data(n)
    pa, nbytes = to_dev(a, dtype)
    pb, _ = to_dev(b, dtype)
    po = rt.malloc(nbytes)
    ct.launch(k, ((n + 1023) // 1024,), pa, pb, po, n)
    rt.synchronize()
    equal = from_dev_raw(po, n, dtype) == pack([a[i] + b[i] for i in range(n)], dtype)
    print(f"  unmasked {dtype} n=4096 逐字节相等={equal}")
    for p in (pa, pb, po):
        rt.free(p)
    return equal


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0), " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE))

    ok16, total16, unsup16 = run_dtype("fp16")
    fp16_ok = (ok16 == total16 and not unsup16)
    print(f"fp16: {ok16}/{total16} 不支持={sorted(set(unsup16))} → 可用={fp16_ok}")

    okb, totalb, unsupb = run_dtype("bf16")
    print(f"bf16: {okb}/{totalb} 不支持={sorted(set(unsupb))}")

    print("=== B. 16 位无掩码（n 为 BLOCK 整数倍）能否绕过缺失的掩码模板 ===")
    fp16_nomask = probe_unmasked("fp16")
    bf16_nomask = probe_unmasked("bf16")
    print(f"  → fp16 无掩码可跑={fp16_nomask}  bf16 无掩码可跑={bf16_nomask}")

    if fp16_ok:
        bench_fp16()
    else:
        print("=== C. fp16 带宽：跳过（fp16 内核在当前链路上编不出来）===")
    print("[end] torch_modules =", len(torchmods()))

    print(f"SUMMARY fp16={ok16}/{total16} bf16={okb}/{totalb} "
          f"unsupported={sorted(set(unsupb))} fp16_nomask={fp16_nomask} bf16_nomask={bf16_nomask}")
    print("DTYPE_VERDICT", "PASS" if fp16_ok else "FP16_UNSUPPORTED")
    return 0 if fp16_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 类型矩阵：fp16 / bf16 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/dtype_matrix.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE171_DONE py_rc=$PYRC"
exit "$PYRC"
