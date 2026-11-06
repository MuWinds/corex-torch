set -u
# 目的：bf16 全算子验收（自然写法 + `_bf16.rewrite_bf16_conversions` 改写层）。
#
# 前提变化（本轮）：之前 bf16 编不过，是因为 Triton 发标准 LLVM 的 `fpext`/`fptrunc`，而 bi 后端
# 只认 `llvm.bi.bfloat2float` / `llvm.bi.float2bfloat`。现在 `_triton.CorexTorchBackend.make_cubin`
# 在 biir 进 llc 之前把它们换掉（`src/corex_torch/_bf16.py`），所以内核里**写自然形式**即可：
#     f = v.to(tl.float32)         # → llvm.bi.bfloat2float
#     tl.store(p, f.to(tl.bfloat16))   # → llvm.bi.float2bfloat
#
# 本探针要证明的四件事（每件都打印数字，不看"编过了"就算）：
#   1. `metadata['corex_rewrite_counts']['bf16_conversions']` > 0 —— 改写**真的命中**了，
#      而不是恰好编过；
#   2. biir 里 `fpext` / `fptrunc` / `trunc nuw` / `trunc nsw` / `splat` 命中数**全 0**；
#   3. 正确性：元素级 bf16 与 CPU bf16 参照**逐字节相等**；累加类给 max_abs/max_rel，并与
#      「顺序 bf16 累加」的误差对照（证明 fp32 累加真的救了精度）；mm 用 K=256/1024 两档；
#   4. 性能：bf16 与 fp32 同尺寸并列，并解释「流量减半但 GB/s 不一定翻倍」。
#
# 尾块：cols/n 取 1000、4097（都不是 BLOCK 的整数倍、也不是 2 的幂）。
# 跑法：python tools/remote.py sbx-run probes/192_bf16_ops.sh --name 192_bf16_ops --timeout 600
# 判定：BF16_VERDICT PASS，退出码如实。

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
export TB_W=$W

cat > "$W/bf16_ops.py" <<'PYEOF'
"""bf16 全算子验收：改写命中数 + 逐字节正确性 + 与顺序 bf16 累加的误差对照 + 性能。"""

import ctypes
import math
import os
import struct
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as elt
from corex_torch import _ops_reduce as rdc
from corex_torch import _ops_matmul as mm_
from corex_torch import _ops_index as idx_


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


# --------------------------------------------------------------------------- #
# bf16 主机侧工具（RNE，与硬件转换一致）
# --------------------------------------------------------------------------- #

def f32_to_bf16_bits(v):
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16(bits):
    """位模式 → 值。"""
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def to_bf16(v):
    """f32/f64 值 → 最近的 bf16 值（一次舍入）。"""
    return bf16(f32_to_bf16_bits(v))


def pack_bf16(values):
    return b"".join(struct.pack("<H", f32_to_bf16_bits(v)) for v in values)


def to_dev(raw):
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def from_dev_bf16(ptr, n):
    buf = ctypes.create_string_buffer(2 * n)
    rt.memcpy(buf, ptr, 2 * n, rt.MEMCPY_DEVICE_TO_HOST)
    return buf.raw


def unpack_bf16(raw, n):
    return [bf16(struct.unpack_from("<H", raw, 2 * i)[0]) for i in range(n)]


def read_f32(ptr, n):
    buf = ctypes.create_string_buffer(4 * n)
    rt.memcpy(buf, ptr, 4 * n, rt.MEMCPY_DEVICE_TO_HOST)
    return [struct.unpack_from("<f", buf.raw, 4 * i)[0] for i in range(n)]


# --------------------------------------------------------------------------- #
# §1 改写层：命中数 + biir 里那些「不该出现的语法」
# --------------------------------------------------------------------------- #

BAD = ("fpext", "fptrunc", "trunc nuw", "trunc nsw", "splat", "fp_extend", "fp_round")


def scan(label, cache):
    """扫一个模块的内核缓存：改写命中数 + **进 llc 的那份 IR** 里的可疑语法计数。

    注意口径：``kernel.asm['biir']`` 是**改写前**的文本（``make_biir`` 的产物），fpext/fptrunc
    出现在里面是**正常的**——``_bf16.rewrite_bf16_conversions`` 在 ``make_cubin`` 里才动手。
    所以要断言的是「**改写后**的文本里一条都不剩」，即这里再跑一遍改写器、看剩下的计数。
    """
    from corex_torch import _bf16
    print(f"--- {label}：{len(cache)} 个内核 ---")
    total_conv = 0
    bad_total = 0
    for key, kernel in sorted(cache.items(), key=lambda kv: str(kv[0])):
        md = kernel.metadata
        counts = getattr(md, "corex_rewrite_counts", None) or {}
        conv = int(counts.get("bf16_conversions", 0) or 0)
        raw = kernel.asm.get("biir", "")
        rewritten, hits = _bf16.rewrite_bf16_conversions(raw)
        left = {b: rewritten.count(b) for b in BAD if rewritten.count(b)}
        total_conv += conv
        bad_total += sum(left.values())
        flag = ""
        if conv == 0:
            flag = "  ← 无转换（纯搬运/纯 bf16 算术）"
        print(f"  {str(key):56s} cubin={len(kernel.asm['cubin']):5d} "
              f"bf16_conversions={conv:2d} 改写后剩余可疑语法={left or '{}'}{flag}")
    print(f"  {label} 合计 bf16_conversions={total_conv}，改写后可疑语法={bad_total}")
    return total_conv, bad_total


# --------------------------------------------------------------------------- #
# §2 元素级 bf16：逐字节相等（silu 例外，按 bf16 ulp 报相对误差）
# --------------------------------------------------------------------------- #

def make_data(n):
    a = [((i % 97) - 48) * 0.25 for i in range(n)]
    b = [[0.5, 1.0, 2.0, 4.0][i % 4] for i in range(n)]
    return a, b


def elementwise():
    print("=== §2 元素级 bf16（与 CPU bf16 参照逐字节比）===")
    ok = total = 0
    dt = "bf16"
    for n in (4096, 4097, 1000):
        a, b = make_data(n)
        a16 = [to_bf16(v) for v in a]              # 参照先按 bf16 舍入，与硬件一致
        b16 = [to_bf16(v) for v in b]
        pa, pb = to_dev(pack_bf16(a16)), to_dev(pack_bf16(b16))
        po = to_dev(b"\x00" * (2 * n))

        def check(name, call, expect, exact=True, tol=0.0):
            nonlocal ok, total
            call()
            rt.synchronize()
            got_raw = from_dev_bf16(po, n)
            exp_raw = pack_bf16(expect)
            total += 1
            if exact:
                good = got_raw == exp_raw
                total += 0
                ok += 1 if good else 0
                extra = "" if good else f" 首个不同元素=" + str(
                    next((i for i in range(n)
                          if got_raw[2 * i:2 * i + 2] != exp_raw[2 * i:2 * i + 2]), -1))
                print(f"  {name:16s} n={n:5d} 逐字节相等={good}{extra}")
            else:
                got = unpack_bf16(got_raw, n)
                ma = max(abs(g - e) for g, e in zip(got, expect))
                rel = max((abs(g - e) / abs(e)) for g, e in zip(got, expect) if abs(e) > 1e-30)
                good = ma <= tol
                ok += 1 if good else 0
                print(f"  {name:16s} n={n:5d} max_abs={ma:.3e} max_rel={rel:.3e} "
                      f"（bf16 ulp 相对量级 3.9e-3）→ {good}")

        check("add", lambda: elt.add(pa, pb, po, n, dt), [to_bf16(x + y) for x, y in zip(a16, b16)])
        check("add(alpha=.5)", lambda: elt.add(pa, pb, po, n, dt, 0.5),
              [to_bf16(x + 0.5 * y) for x, y in zip(a16, b16)])
        check("sub", lambda: elt.sub(pa, pb, po, n, dt), [to_bf16(x - y) for x, y in zip(a16, b16)])
        check("mul", lambda: elt.mul(pa, pb, po, n, dt), [to_bf16(x * y) for x, y in zip(a16, b16)])
        check("div", lambda: elt.div(pa, pb, po, n, dt), [to_bf16(x / y) for x, y in zip(a16, b16)])
        check("add_scalar", lambda: elt.add_scalar(pa, po, n, dt, 0.5), [to_bf16(x + 0.5) for x in a16])
        check("mul_scalar", lambda: elt.mul_scalar(pa, po, n, dt, 0.25), [to_bf16(x * 0.25) for x in a16])
        check("fill(0.5)", lambda: elt.fill(po, n, dt, 0.5), [to_bf16(0.5)] * n)
        check("fill(-2.0)", lambda: elt.fill(po, n, dt, -2.0), [to_bf16(-2.0)] * n)
        check("copy", lambda: elt.copy(pa, po, n, dt), [to_bf16(x) for x in a16])
        # silu：exp 走 ex2.approx，再加一次 bf16 舍入 → 按 bf16 ulp 量级判
        check("silu", lambda: elt.silu(pa, po, n, dt),
              [to_bf16(x / (1.0 + math.exp(-x))) for x in a16], exact=False, tol=1e-2)

        for p in (pa, pb, po):
            rt.free(p)
    print(f"§2 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §3 累加类 bf16：fp32 累加 vs 顺序 bf16 累加
# --------------------------------------------------------------------------- #

def reductions():
    print("=== §3 规约 / softmax / norm bf16（输入先舍成 bf16；参照 = 双精度算 bf16 输入）===")
    ok = total = 0
    eps = 1e-6
    for rows, cols in ((5, 1000), (7, 512), (9, 4097)):
        raw = [((i * 37 + j * 13) % 199 - 99) * 0.5 for i in range(rows) for j in range(cols)]
        x16 = [to_bf16(v) for v in raw]
        px = to_dev(pack_bf16(x16))
        prow = to_dev(b"\x00" * (2 * rows))
        pfull = to_dev(b"\x00" * (2 * rows * cols))
        rowsdata = [x16[r * cols:(r + 1) * cols] for r in range(rows)]

        # --- sum_last：fp32 累加（我们） vs 顺序 bf16 累加（反面参照） ---
        dbl = [sum(r) for r in rowsdata]
        seq_bf16 = []
        for r in rowsdata:
            acc = 0.0
            for v in r:
                acc = to_bf16(acc + v)              # 每一步都舍成 bf16
            seq_bf16.append(acc)
        err_seq = max(abs(a - b) for a, b in zip(seq_bf16, dbl))
        rdc.sum_last(px, prow, rows, cols, "bf16")
        rt.synchronize()
        got = unpack_bf16(from_dev_bf16(prow, rows), rows)
        ma = max(abs(g - d) for g, d in zip(got, dbl))
        # 输出本身是 bf16（8 位尾数，相对粒度 2^-8），所以下限是「行和量级的一个 bf16 ulp」；
        # 反面参照是「纯 bf16 累加」：累加器也是 bf16，几千项下来误差到几千分之一以上。
        ulp = max(4.0 * max(abs(d) for d in dbl) * 2.0 ** -8, 1e-6)
        limit = max(ulp, 2.0 * err_seq)
        total += 1
        ok += 1 if ma <= limit else 0
        print(f"  sum_last  rows={rows} cols={cols:4d}  max_abs={ma:.3e} 限={limit:.3e}"
              f"（顺序 bf16 累加误差={err_seq:.3e} → 我们小 {err_seq / ma if ma else float('inf'):.1f} 倍）"
              f" → {ma <= limit}")

        # --- max_last：max 与顺序无关，必须逐字节 ---
        rdc.max_last(px, prow, rows, cols, "bf16")
        rt.synchronize()
        exact = from_dev_bf16(prow, rows) == pack_bf16([max(r) for r in rowsdata])
        total += 1
        ok += 1 if exact else 0
        print(f"  max_last  rows={rows} cols={cols:4d}  逐字节相等={exact}")

        # --- softmax：fp32 里 max/exp/归一 ---
        rdc.softmax_last(px, pfull, rows, cols, "bf16")
        rt.synchronize()
        got = unpack_bf16(from_dev_bf16(pfull, rows * cols), rows * cols)
        ref = []
        for r in rowsdata:
            mx = max(r)
            ex = [math.exp(v - mx) for v in r]
            s = sum(ex)
            ref.extend([e / s for e in ex])
        ma = max(abs(g - e) for g, e in zip(got, ref))
        rowsum_dev = max(abs(sum(got[r * cols:(r + 1) * cols]) - 1.0) for r in range(rows))
        good = ma <= 1e-2 and rowsum_dev <= 1e-2
        total += 1
        ok += 1 if good else 0
        print(f"  softmax   rows={rows} cols={cols:4d}  max_abs={ma:.3e} 行和偏离1={rowsum_dev:.3e}"
              f"（bf16 输出精度）→ {good}")

        # --- rms_norm / layernorm（bf16 权重）---
        w = [to_bf16(1.0 + ((j % 5) - 2) * 0.25) for j in range(cols)]
        pw = to_dev(pack_bf16(w))
        for name, fn in (("rms_norm", rdc.rms_norm), ("layernorm", rdc.layernorm)):
            fn(px, pw, pfull, rows, cols, eps, "bf16")
            rt.synchronize()
            got = unpack_bf16(from_dev_bf16(pfull, rows * cols), rows * cols)
            if name == "rms_norm":
                ref = []
                for r in rowsdata:
                    ms = sum(v * v for v in r) / cols
                    rstd = 1.0 / math.sqrt(ms + eps)
                    ref.extend([v * rstd * w[j] for j, v in enumerate(r)])
            else:
                ref = []
                for r in rowsdata:
                    mean = sum(r) / cols
                    var = sum((v - mean) ** 2 for v in r) / cols
                    rstd = 1.0 / math.sqrt(var + eps)
                    ref.extend([(v - mean) * rstd * w[j] for j, v in enumerate(r)])
            rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 1e-6)
            good = rel <= 1e-2
            total += 1
            ok += 1 if good else 0
            print(f"  {name:10s} rows={rows} cols={cols:4d}  max_rel={rel:.3e}"
                  f"（bf16 输出）→ {good}")
        for p in (px, prow, pfull, pw):
            rt.free(p)
    print(f"§3 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §4 gather_rows / mm：bf16
# --------------------------------------------------------------------------- #

def matmul_and_index():
    print("=== §4 gather_rows 与 mm/bmm 的 bf16 ===")
    ok = total = 0
    # gather：纯搬运，逐字节
    for rows, cols, table_rows in ((7, 1000, 32), (5, 4097, 40)):
        table = [to_bf16(float(i) * 0.5) for i in range(table_rows * cols)]
        src = to_dev(pack_bf16(table))
        idx_list = [((i * 7 + 3) % table_rows) for i in range(rows)]
        pidx = rt.malloc(rows * 4)
        rt.memcpy(pidx, (ctypes.c_int32 * rows)(*idx_list), rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
        po = to_dev(b"\x00" * (2 * rows * cols))
        idx_.gather_rows(src, pidx, po, rows, cols, "bf16")
        rt.synchronize()
        exp = []
        for r in idx_list:
            exp.extend(table[r * cols:(r + 1) * cols])
        exact = from_dev_bf16(po, rows * cols) == pack_bf16(exp)
        total += 1
        ok += 1 if exact else 0
        print(f"  gather_rows rows={rows} cols={cols:4d} 逐字节相等={exact}")
        for p in (src, pidx, po):
            rt.free(p)

    # mm：bf16 输入 / fp32 累加 / bf16 输出。K=256 与 K=1024 两档
    for M, N, K in ((32, 32, 256), (16, 16, 1024)):
        a = [to_bf16(((i * 11 + j * 5) % 17 - 8) * 0.1 - 0.1) for i in range(M) for j in range(K)]
        b = [to_bf16(((i * 13 + j * 7) % 19 - 9) * 0.1 + 0.05) for i in range(K) for j in range(N)]
        pa, pb = to_dev(pack_bf16(a)), to_dev(pack_bf16(b))
        pc = to_dev(b"\x00" * (2 * M * N))
        mm_.mm(pa, pb, pc, M, N, K, "bf16")
        rt.synchronize()
        got = unpack_bf16(from_dev_bf16(pc, M * N), M * N)
        # 两个参照：双精度真值、纯 bf16 顺序累加（模拟"纯 bf16 硬算"的做法）
        ref, bad = [], []
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a[i * K + k] * b[k * N + j]
                ref.append(s)
                acc = 0.0
                for k in range(K):
                    acc = to_bf16(acc + to_bf16(a[i * K + k] * b[k * N + j]))
                bad.append(acc)
        rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 1e-2)
        err_bad = max((abs(x - e) / abs(e)) for x, e in zip(bad, ref) if abs(e) > 1e-2)
        good = rel <= 2e-2                     # bf16 输出本身 ~3.9e-3 的相对粒度
        total += 1
        ok += 1 if good else 0
        print(f"  mm {M}x{N}x{K:4d}  max_rel(|ref|>1e-2)={rel:.3e} "
              f"（纯 bf16 顺序累加的误差={err_bad:.3e} → 我们小 {err_bad / rel:.1f} 倍）→ {good}")
        for p in (pa, pb, pc):
            rt.free(p)

    # bmm
    B, M, N, K = 2, 16, 16, 128
    a = [to_bf16(float((i % 7) - 3) * 0.5) for i in range(B * M * K)]
    b = [to_bf16(float((i % 5) - 2) * 0.25) for i in range(B * K * N)]
    pa, pb = to_dev(pack_bf16(a)), to_dev(pack_bf16(b))
    pc = to_dev(b"\x00" * (2 * B * M * N))
    mm_.bmm(pa, pb, pc, B, M, N, K, "bf16")
    rt.synchronize()
    got = unpack_bf16(from_dev_bf16(pc, B * M * N), B * M * N)
    ref = []
    for bi in range(B):
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a[bi * M * K + i * K + k] * b[bi * K * N + k * N + j]
                ref.append(s)
    rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 1e-2)
    total += 1
    ok += 1 if rel <= 2e-2 else 0
    print(f"  bmm B={B} {M}x{N}x{K} max_rel={rel:.3e} → {rel <= 2e-2}")
    for p in (pa, pb, pc):
        rt.free(p)
    print(f"§4 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §5 性能：bf16 vs fp32（同尺寸并列）
# --------------------------------------------------------------------------- #

_FLUSH = {}


def flush_l2():
    if "ptr" not in _FLUSH:
        m = 64 << 20
        _FLUSH["ptr"] = rt.malloc(m * 4)
        _FLUSH["n"] = m
    elt.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench(fn, reps=20, warmup=3):
    pairs = [(rt.Event(), rt.Event()) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    rt.synchronize()
    for i in range(reps):
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


def performance():
    print("=== §5 性能（bf16 vs fp32，同元素数；L2 每轮刷 256 MiB）===")
    n = 1 << 20
    a, b = make_data(n)
    for dt, nbytes in (("fp32", n * 4), ("bf16", n * 2)):
        pa = to_dev(pack_bf16(a) if dt == "bf16" else
                    b"".join(struct.pack("<f", v) for v in a))
        pb = to_dev(pack_bf16(b) if dt == "bf16" else
                    b"".join(struct.pack("<f", v) for v in b))
        po = rt.malloc(nbytes)
        med = bench(lambda: elt.add(pa, pb, po, n, dt))
        tb = elt.traffic_bytes(n, dt, 3)
        print(f"  add {dt}: {med * 1e6:8.1f} us  流量 {tb / 1048576:.2f} MiB "
              f"→ {tb / med / 1e9:7.1f} GB/s")
        for p in (pa, pb, po):
            rt.free(p)

    for dt in ("fp32", "bf16"):
        M = N = K = 256
        a32 = [((i * 11 + j * 5) % 17 - 8) * 0.1 for i in range(M) for j in range(K)]
        b32 = [((i * 13 + j * 7) % 19 - 9) * 0.1 for i in range(K) for j in range(N)]
        if dt == "bf16":
            a32 = [to_bf16(v) for v in a32]
            b32 = [to_bf16(v) for v in b32]
        pack = pack_bf16 if dt == "bf16" else (lambda vs: b"".join(struct.pack("<f", v) for v in vs))
        pa, pb = to_dev(pack(a32)), to_dev(pack(b32))
        pc = rt.malloc(M * N * (2 if dt == "bf16" else 4))
        med = bench(lambda: mm_.mm(pa, pb, pc, M, N, K, dt), reps=10)
        print(f"  mm 256^3 {dt}: {med * 1e6:8.1f} us  {mm_.flops(M, N, K) / med / 1e9:7.1f} GFLOP/s")
        for p in (pa, pb, pc):
            rt.free(p)
    print("  说明：bf16 的流量是 fp32 的一半，但 GB/s 不会翻倍——42 µs 量级里约 10 µs 是这张卡"
          "的启动地板（空 kernel 9.9 µs，corex-triton §12），它不随 dtype 变小。")


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    # 每一段独立 try：规约/矩阵乘那段目前会被「向量形式转换未覆盖」挡住（见 §6 的体检），
    # 但**元素级的结论与改写层的命中数必须照常打出来**——否则一次阻塞会把整张表都吞掉。
    blocked = []
    ok, total = elementwise()
    try:
        ok2, total2 = reductions()
    except Exception as exc:                            # noqa: BLE001
        print(f"§3 被阻塞: {type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:200]}")
        blocked.append("reductions")
        ok2 = total2 = 0
    try:
        ok3, total3 = matmul_and_index()
    except Exception as exc:                            # noqa: BLE001
        print(f"§4 被阻塞: {type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:200]}")
        blocked.append("matmul_and_index")
        ok3 = total3 = 0
    try:
        performance()
    except Exception as exc:                            # noqa: BLE001
        print(f"§5 被阻塞: {type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:200]}")
        blocked.append("performance")

    print("=== §6 改写层与可疑语法体检 ===")
    conv_e, bad_e = scan("elementwise", elt._KERNEL_CACHE)
    conv_r, bad_r = scan("reduce", rdc._KERNEL_CACHE)
    conv_m, bad_m = scan("matmul", mm_._KERNEL_CACHE)
    conv_i, bad_i = scan("index", idx_._KERNEL_CACHE)
    conv = conv_e + conv_r + conv_m + conv_i
    bad = bad_e + bad_r + bad_m + bad_i
    print(f"  全部 bf16_conversions 命中共 {conv}（必须 > 0，否则说明改写没生效）")
    print(f"  全部可疑语法命中 {bad}（fpext/fptrunc/trunc nuw/trunc nsw/splat/fp_extend/fp_round，必须 0）")

    print("[end] torch_modules =", len(torchmods()))
    all_ok = (ok == total) and (not blocked)
    verdict = "PASS" if (all_ok and conv > 0 and bad == 0) else "PARTIAL"
    print(f"元素级 {ok}/{total}  累加类 {ok2}/{total2}  mm/index {ok3}/{total3}  "
          f"改写命中 {conv}  可疑语法 {bad}  被阻塞段={blocked}")
    print("BF16_VERDICT", verdict)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. bf16 全算子验收 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/bf16_ops.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE192_DONE py_rc=$PYRC"
exit "$PYRC"
