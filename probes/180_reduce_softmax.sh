set -u
# 目的：验收规约类算子（`_ops_reduce.py`）：行内 sum/mean/max、全量 sum、逐行 softmax。
#   1. §0 顺带把三个新模块（reduce / matmul / index）的**每个内核都编一遍**，一次把
#      「谁的哪条路编不出来」暴露干净，省掉三轮排队；
#   2. 正确性：与 CPU **双精度**参照比，并同时给出「CPU 顺序 fp32 累加 vs 双精度」的差值
#      —— 后者就是「fp32 累加顺序不同」的自然量级。判定不看某个魔数，而是要求
#      我们的误差 ≤ 8 × 这个量级（理由见 docs/04 §7.4）；
#   3. 尾块/非整除：cols 取 512 / 1000 / 4097（后两个都不是 2 的幂，也不是 BLOCK 的整数倍）；
#   4. 性能：softmax / sum_last / sum_all 的 GB/s（每轮刷 256 MiB L2，事件计时）。
#
# 载荷：`work/ops/payload.tgz`（corex-torch/src/corex_torch + corex-triton/src/triton_corex）。
# 跑法：python tools/remote.py sbx-run probes/180_reduce_softmax.sh --name 180_reduce --timeout 600
# 判定：REDUCE_VERDICT PASS，退出码如实。

PY=$HOME/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=$HOME/coffin-work/base/sandbox
W=$S/work/ops
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname) DATE=$(date -Is) PY=$PY"
mkdir -p "$W"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES，不覆盖）==="
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
ls "$W/corex-torch/src/corex_torch/_ops_reduce.py" \
   "$W/corex-torch/src/corex_torch/_ops_matmul.py" \
   "$W/corex-torch/src/corex_torch/_ops_index.py" || exit 67
export PYTHONPATH=$W/corex-torch/src
export COREX_TRITON_SRC=$W/corex-triton/src
export TB_W=$W
# 让 corex-triton 把改写后的 biir 落盘：编译失败时探针会把 llc 指的那一行打出来。
export COREX_DUMP_IR=$W/biir_dump.ll

cat > "$W/reduce_check.py" <<'PYEOF'
"""规约类算子：编译烟测 + 正确性（双精度参照 + fp32 累加顺序量级）+ 带宽。"""

import ctypes
import math
import os
import struct
import sys
import time

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as elt
from corex_torch import _ops_reduce as rdc
from corex_torch import _ops_matmul as mm_
from corex_torch import _ops_index as idx_

#: 探针自己的工作目录（dump 与日志都写这里）。
W = os.environ.get("TB_W", ".")


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


# --------------------------------------------------------------------------- #
# 主机侧打包 / 解包 / fp32 舍入模拟
# --------------------------------------------------------------------------- #

FMT = {"fp32": "<f", "fp16": "<e"}


def round_f32(v):
    """把一个 double 舍到 fp32 再还原（模拟一步一步 fp32 累加）。"""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def pack(values, dtype):
    if dtype == "fp16":
        return b"".join(struct.pack("<e", v) for v in values)
    return b"".join(struct.pack("<f", v) for v in values)


def unpack(raw, n, dtype):
    fmt = FMT[dtype]
    size = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, raw, size * i)[0] for i in range(n)]


def to_dev(values, dtype):
    raw = pack(values, dtype)
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr, len(raw)


def from_dev(ptr, n, dtype):
    size = elt.itemsize(dtype) * max(n, 1)
    buf = ctypes.create_string_buffer(size)
    rt.memcpy(buf, ptr, size, rt.MEMCPY_DEVICE_TO_HOST)
    return unpack(buf.raw, n, dtype)


def make_matrix(rows, cols):
    """取 fp32 精确可表示的值：|x| ≤ 50，步长 0.5。"""
    return [float(((i * 37 + j * 13) % 199) - 99) * 0.5
            for i in range(rows) for j in range(cols)]


# --------------------------------------------------------------------------- #
# §0 三个模块的编译烟测
#
# 每个缓冲都按**真实形状**精确分配：把 rows×cols 的输出写进一个 16 字节的缓冲会越界写，
# 在 GPU 上那不是崩溃而是「静默踩坏别人的显存」，事后表现为莫名其妙的数值错。
# --------------------------------------------------------------------------- #

def dump_diag(exc):
    """编译失败时把 llc 的完整报错与落盘 biir 里出错的那一行打出来。

    只打印 `str(exc)` 的第一行是不够的：厂商 llc 的语法错只说「expected …」，
    必须看它指的 m.ll 第几行、以及我们改写后的 biir 对应那一行长什么样。
    """
    text = str(exc).replace("\n", " | ")
    print(f"       llc 全文: {text[:900]}")
    import glob
    import os
    import re
    hits = sorted(glob.glob(os.path.join(W, "biir_dump*.ll")), key=os.path.getmtime)
    if not hits:
        print("       （没有 biir dump：编译没走到 cubin 阶段）")
        return
    m = re.search(r"m\.ll:(\d+):(\d+)", text)
    lines = open(hits[-1], encoding="utf-8", errors="replace").read().splitlines()
    print(f"       dump={os.path.basename(hits[-1])} 共 {len(lines)} 行")
    if m:
        ln = int(m.group(1))
        for k in range(max(0, ln - 3), min(len(lines), ln + 2)):
            mark = ">>" if k == ln - 1 else "  "
            print(f"       {mark} {k + 1:4d}: {lines[k][:200]}")
    else:
        for k, ln_text in enumerate(lines[:40]):
            print(f"          {k + 1:4d}: {ln_text[:200]}")


def smoke():
    """把每个模块的每个内核各编一次并跑一发极小规模，一次暴露所有编译问题。"""
    print("=== §0 编译烟测（reduce / matmul / index 全内核）===")
    ok = fail = 0
    rows, cols = 4, 64
    n = rows * cols
    # 源表：4×64 的 fp32；下标表单独构造（拿浮点数据当 int32 下标会读到天知道的地方）
    src, _ = to_dev([float(i) * 0.5 for i in range(n)], "fp32")
    idx32 = (ctypes.c_int32 * rows)(0, 1, 2, 3)
    idx64 = (ctypes.c_int64 * rows)(0, 1, 2, 3)
    pidx32 = rt.malloc(rows * 4)
    pidx64 = rt.malloc(rows * 8)
    rt.memcpy(pidx32, idx32, rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
    rt.memcpy(pidx64, idx64, rows * 8, rt.MEMCPY_HOST_TO_DEVICE)
    prow = rt.malloc(rows * 4)              # 每行一个结果（rows 个 fp32）
    pfull = rt.malloc(n * 4)                # rows×cols 的整块输出
    p1 = rt.malloc(4)                       # 单个标量结果
    pw = rt.malloc(cols * 4)                # norm 的权重（长度 cols）
    p16 = rt.malloc(n * 2)                  # fp16 整块缓冲
    p16row = rt.malloc(rows * 2)

    cases = [
        ("sum_last", lambda: rdc.sum_last(src, prow, rows, cols, "fp32")),
        ("mean_last", lambda: rdc.mean_last(src, prow, rows, cols, "fp32")),
        ("max_last", lambda: rdc.max_last(src, prow, rows, cols, "fp32")),
        ("sum_all", lambda: rdc.sum_all(src, p1, n, "fp32")),
        ("softmax_last", lambda: rdc.softmax_last(src, pfull, rows, cols, "fp32")),
        ("rms_norm", lambda: rdc.rms_norm(src, pw, pfull, rows, cols, 1e-6, "fp32")),
        ("layernorm", lambda: rdc.layernorm(src, pw, pfull, rows, cols, 1e-6, "fp32")),
        ("sum_last fp16", lambda: rdc.sum_last(p16, p16row, rows, cols, "fp16")),
        ("softmax fp16", lambda: rdc.softmax_last(p16, p16, rows, cols, "fp16")),
        ("mm 8x8x8", lambda: mm_.mm(src, src, pfull, 8, 8, 8, "fp32")),
        ("bmm 2x4x8x8", lambda: mm_.bmm(src, src, pfull, 2, 4, 8, 8, "fp32")),
        ("mm 17x19x23（非整块）", lambda: mm_.mm(src, src, pfull, 17, 19, 23, "fp32")),
        ("gather_rows", lambda: idx_.gather_rows(src, pidx32, pfull, rows, cols, "fp32")),
        ("gather_rows i64", lambda: idx_.gather_rows(src, pidx64, pfull, rows, cols, "fp32",
                                                    idx_dtype="i64")),
    ]
    for name, fn in cases:
        try:
            fn()
            rt.synchronize()
            print(f"  OK   {name}")
            ok += 1
        except Exception as exc:                        # noqa: BLE001 - 失败也是结论
            print(f"  FAIL {name}: {type(exc).__name__}")
            dump_diag(exc)
            fail += 1
    for p in (src, pidx32, pidx64, prow, pfull, p1, pw, p16, p16row):
        rt.free(p)
    print(f"  烟测 {ok}/{ok + fail}")
    return fail == 0


# --------------------------------------------------------------------------- #
# §1 正确性
# --------------------------------------------------------------------------- #

def correctness():
    print("=== §1 正确性（fp32；参照 = CPU 双精度，误差量级用「顺序 fp32 累加」标定）===")
    ok = total = 0

    for rows, cols in ((7, 512), (5, 1000), (9, 4097)):
        data = make_matrix(rows, cols)
        pa, _ = to_dev(data, "fp32")
        prow = rt.malloc(rows * 4)
        pfull = rt.malloc(rows * cols * 4)
        rowdata = [data[r * cols:(r + 1) * cols] for r in range(rows)]

        # 参照：双精度和 + 「一步步 fp32 累加」的和（后者就是 fp32 累加顺序误差的尺度）
        s_dbl = [sum(row) for row in rowdata]
        s_f32 = []
        for row in rowdata:
            acc = 0.0
            for v in row:
                acc = round_f32(acc + v)
            s_f32.append(acc)
        gap = max(abs(a - b) for a, b in zip(s_f32, s_dbl))
        limit = 8.0 * gap + 1e-6 * max(1.0, max(abs(x) for x in s_dbl))

        # --- sum_last ---
        rdc.sum_last(pa, prow, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(prow, rows, "fp32")
        ma = max(abs(g - r) for g, r in zip(got, s_dbl))
        total += 1
        ok += 1 if ma <= limit else 0
        print(f"  sum_last  rows={rows:2d} cols={cols:4d}  max_abs={ma:.3e} "
              f"限={limit:.3e}（顺序fp32误差={gap:.3e}）→ {ma <= limit}")

        # --- mean_last ---（误差按 sum 的尺子除以 cols）
        rdc.mean_last(pa, prow, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(prow, rows, "fp32")
        m_dbl = [s / cols for s in s_dbl]
        ma = max(abs(g - r) for g, r in zip(got, m_dbl))
        limit_m = limit / cols
        total += 1
        ok += 1 if ma <= limit_m else 0
        print(f"  mean_last rows={rows:2d} cols={cols:4d}  max_abs={ma:.3e} "
              f"限={limit_m:.3e} → {ma <= limit_m}")

        # --- max_last ---（与累加顺序无关，必须逐字节精确）
        rdc.max_last(pa, prow, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(prow, rows, "fp32")
        m_ref = [max(row) for row in rowdata]
        exact = pack(got, "fp32") == pack(m_ref, "fp32")
        total += 1
        ok += 1 if exact else 0
        print(f"  max_last  rows={rows:2d} cols={cols:4d}  逐字节相等={exact}")

        # --- softmax_last ---（ex2.approx：误差单列，不混进逐字节口径）
        rdc.softmax_last(pa, pfull, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(pfull, rows * cols, "fp32")
        ref = []
        for row in rowdata:
            mx = max(row)
            ex = [math.exp(v - mx) for v in row]
            s = sum(ex)
            ref.extend([e / s for e in ex])
        ma = max(abs(g - r) for g, r in zip(got, ref))
        row_sums = [sum(got[r * cols:(r + 1) * cols]) for r in range(rows)]
        sum_dev = max(abs(s - 1.0) for s in row_sums)
        good = ma <= 1e-6 and sum_dev <= 1e-6
        total += 1
        ok += 1 if good else 0
        print(f"  softmax   rows={rows:2d} cols={cols:4d}  max_abs={ma:.3e} "
              f"行和偏离1={sum_dev:.3e} → {good}")

        # --- softmax 原地（out == x）---
        rdc.softmax_last(pa, pa, rows, cols, "fp32")
        rt.synchronize()
        inplace = from_dev(pa, rows * cols, "fp32")
        same = pack(inplace, "fp32") == pack(got, "fp32")
        total += 1
        ok += 1 if same else 0
        print(f"  softmax 原地 out==x      与 out-of-place 逐字节相同={same}")

        for p in (pa, prow, pfull):
            rt.free(p)

    # --- sum_all ---
    for n in (4096, 100000, 1 << 20):
        vals = [float(((i * 29) % 101) - 50) * 0.5 for i in range(n)]
        pa, _ = to_dev(vals, "fp32")
        p1 = rt.malloc(4)
        dbl = sum(vals)
        acc = 0.0
        for v in vals:
            acc = round_f32(acc + v)
        gap = abs(acc - dbl)
        rdc.sum_all(pa, p1, n, "fp32")
        rt.synchronize()
        got = from_dev(p1, 1, "fp32")[0]
        ma = abs(got - dbl)
        limit = 8.0 * gap + 1e-6 * max(1.0, abs(dbl))
        total += 1
        ok += 1 if ma <= limit else 0
        print(f"  sum_all   n={n:8d}          max_abs={ma:.3e} 限={limit:.3e}"
              f"（顺序fp32误差={gap:.3e}）→ {ma <= limit}  和={got:.6f}")
        rt.free(pa)
        rt.free(p1)

    # --- sum_all 的 G 上限：G 必须同时 ≤ _MAX_PARTS 与 BLOCK，否则第二段会漏部分和 ---
    # block=256 时 G 被夹到 256、chunk 涨到 4096（循环 16 轮），正好覆盖「夹住 + 多轮循环」。
    for n, blk in ((1 << 20, 256), (1 << 20, 4096)):
        vals = [float(((i * 29) % 101) - 50) * 0.5 for i in range(n)]
        pa, _ = to_dev(vals, "fp32")
        p1 = rt.malloc(4)
        dbl = sum(vals)
        rdc.sum_all(pa, p1, n, "fp32", block=blk)
        rt.synchronize()
        got = from_dev(p1, 1, "fp32")[0]
        # G 被夹住时循环变长，累加顺序也变；用「相对双精度的误差 ≤ 1e-5×量级」判
        ma = abs(got - dbl)
        good = ma <= 1e-5 * max(1.0, abs(dbl))
        total += 1
        ok += 1 if good else 0
        print(f"  sum_all   n={n:8d} block={blk:4d}  max_abs={ma:.3e}（G≤BLOCK 的回归）"
              f"→ {good}")
        rt.free(pa)
        rt.free(p1)

    # --- 非 dyadic 数据：0.1 的倍数会让 fp32 累加真的产生舍入 ---
    # 上面那组用 0.5 的倍数，和恰好是精确的（max_abs 恒 0），只能证明「算法对」，
    # 证明不了「累加顺序的误差在合理量级」。这一组专门制造舍入。
    for rows, cols in ((9, 4097), (7, 1000)):
        data = [0.1 * (((i * 13 + j * 7) % 97) - 48) for i in range(rows) for j in range(cols)]
        pa, _ = to_dev(data, "fp32")
        prow = rt.malloc(rows * 4)
        rowdata = [data[r * cols:(r + 1) * cols] for r in range(rows)]
        s_dbl = [sum(row) for row in rowdata]
        s_f32 = []
        for row in rowdata:
            acc = 0.0
            for v in row:
                acc = round_f32(acc + v)
            s_f32.append(acc)
        gap = max(abs(a - b) for a, b in zip(s_f32, s_dbl))
        limit = 8.0 * gap + 1e-6 * max(1.0, max(abs(x) for x in s_dbl))
        rdc.sum_last(pa, prow, rows, cols, "fp32")
        rt.synchronize()
        got = from_dev(prow, rows, "fp32")
        ma = max(abs(g - r) for g, r in zip(got, s_dbl))
        total += 1
        ok += 1 if ma <= limit else 0
        print(f"  sum_last(非dyadic) cols={cols:4d}  max_abs={ma:.3e} 限={limit:.3e}"
              f"（顺序fp32误差={gap:.3e}）→ {ma <= limit}")
        rt.free(pa)
        rt.free(prow)

    print(f"§1 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §3 bf16：必须显式 NotImplementedError，绝不静默降精度
# --------------------------------------------------------------------------- #

def bf16_unsupported():
    """每个算子拿 bf16 调一次，确认抛的是 `NotImplementedError` 且带 llc 原文。"""
    print("=== §3 bf16 分支（厂商 llc：bf16 = fp_round 选不出指令）===")
    ok = total = 0
    rows, cols = 4, 64
    n = rows * cols
    src = rt.malloc(n * 2)          # bf16 元素占 2 字节
    dst = rt.malloc(n * 2)
    row = rt.malloc(rows * 4)
    p1 = rt.malloc(4)
    # 下标表单独构造：拿浮点缓冲当下标会读到天知道的地方
    pidx = rt.malloc(rows * 4)
    rt.memcpy(pidx, (ctypes.c_int32 * rows)(0, 1, 2, 3), rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
    cases = [
        ("sum_last", lambda: rdc.sum_last(src, row, rows, cols, "bf16")),
        ("softmax_last", lambda: rdc.softmax_last(src, dst, rows, cols, "bf16")),
        ("rms_norm", lambda: rdc.rms_norm(src, src, dst, rows, cols, 1e-6, "bf16")),
        ("mm", lambda: mm_.mm(src, src, dst, 8, 8, 8, "bf16")),
        ("gather_rows", lambda: idx_.gather_rows(src, pidx, dst, rows, cols, "bf16")),
    ]
    for name, fn in cases:
        total += 1
        try:
            fn()
            rt.synchronize()
            # 编得过就说明这条路不经过收窄（例如纯搬运），那也不算违背契约
            print(f"  {name:13s} 编得过并跑通（不经过 fp32→bf16 收窄）")
            ok += 1
        except NotImplementedError as exc:
            first = str(exc).replace("\n", " ")[:180]
            print(f"  {name:13s} NotImplementedError OK  {first}")
            ok += 1
        except Exception as exc:                        # noqa: BLE001
            print(f"  {name:13s} 抛了 {type(exc).__name__}（应当是 NotImplementedError）: "
                  f"{str(exc).splitlines()[0][:160]}")
    print(f"§3 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §2 带宽
# --------------------------------------------------------------------------- #

_FLUSH = {}


def flush_l2():
    """每轮先写 256 MiB 顶掉 L2（32 MiB）；否则 4 MiB 的工作集全在 L2 里，量成 L2 带宽。"""
    if "ptr" not in _FLUSH:
        m = 64 << 20
        _FLUSH["ptr"] = rt.malloc(m * 4)
        _FLUSH["n"] = m
    elt.fill(_FLUSH["ptr"], _FLUSH["n"], "fp32", 0.0)


def bench(fn, reps=20, warmup=3):
    """事件计时（中位数），每轮刷 L2；返回秒。"""
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


def bandwidth():
    print("=== §2 带宽（1<<20 元素的等价规模 + L2 刷）===")
    rows, cols = 1024, 1000
    data = make_matrix(rows, cols)
    pa, _ = to_dev(data, "fp32")
    po = rt.malloc(rows * 4)
    pfull = rt.malloc(rows * cols * 4)
    nbytes = rows * cols * 4

    # sum_last / mean_last / max_last：读 rows*cols，写 rows
    for name, fn in (("sum_last", lambda: rdc.sum_last(pa, po, rows, cols, "fp32")),
                     ("mean_last", lambda: rdc.mean_last(pa, po, rows, cols, "fp32")),
                     ("max_last", lambda: rdc.max_last(pa, po, rows, cols, "fp32"))):
        dt = bench(fn)
        tb = nbytes + rows * 4
        print(f"  {name:12s} rows={rows} cols={cols}  {dt * 1e6:8.1f} us  "
              f"读 {nbytes / 1048576:.2f} MiB → {tb / dt / 1e9:7.1f} GB/s")

    # softmax：有效流量按「读 1 写 1」；实现读 3 遍（行 4 KB，L1/L2 命中）
    dt = bench(lambda: rdc.softmax_last(pa, pfull, rows, cols, "fp32"))
    tb = 2 * nbytes
    print(f"  {'softmax':12s} rows={rows} cols={cols}  {dt * 1e6:8.1f} us  "
          f"读+写 {tb / 1048576:.2f} MiB → {tb / dt / 1e9:7.1f} GB/s"
          f"（源码读 3 遍，DRAM 上仍 1 遍）")

    # sum_all：全量
    for n in (1 << 20, 1 << 22):
        vals = [float(((i * 29) % 101) - 50) * 0.5 for i in range(n)]
        p, _ = to_dev(vals, "fp32")
        out = rt.malloc(4)
        dt = bench(lambda: rdc.sum_all(p, out, n, "fp32"))
        tb = n * 4
        print(f"  {'sum_all':12s} n={n:8d}          {dt * 1e6:8.1f} us  "
              f"读 {tb / 1048576:.2f} MiB → {tb / dt / 1e9:7.1f} GB/s")
        rt.free(p)
        rt.free(out)

    for p in (pa, po, pfull):
        rt.free(p)
    print("  cache:", rdc.cache_info()["kernels"], "个内核已编")


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0), " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE),
          " BLOCK =", rdc.DEFAULT_BLOCK, " CHUNK =", rdc.DEFAULT_CHUNK)
    smoke_ok = smoke()
    ok, total = correctness()
    ok_bf, total_bf = bf16_unsupported()
    bandwidth()
    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if (smoke_ok and ok == total and ok_bf == total_bf) else "FAIL"
    print(f"烟测={smoke_ok} 正确性={ok}/{total} bf16契约={ok_bf}/{total_bf}")
    print("REDUCE_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 规约 + softmax：烟测 / 正确性 / 带宽 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/reduce_check.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE180_DONE py_rc=$PYRC"
exit "$PYRC"
