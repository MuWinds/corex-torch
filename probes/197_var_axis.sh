set -u
# 目的：最后一批内核的真机验收。
#
#   1. row_var / row_std —— 两趟（先均值、再平方差），并**用数据证明**一趟公式
#      `sum(x^2) - n*mean^2` 会垮：构造「均值远大于标准差」的行（x ≈ 1e6，方差 ~0.16），
#      一处是 fp32 累加的一趟（真实内核会得到的数），一处是双精度的一趟（公式本身的误差），
#      两处都和我们的两趟结果对照。
#   2. sum_axis / mean_axis —— 单个中间维归约，逻辑形状 (outer, red, inner)，inner 连续。
#      多维组合仍由上层显式报错，本批不做。
#   3. （可选）masked_fill 的广播 mask —— 本批未做，理由写在 docs §10。
#
# 口径：求和类逐字节（dyadic 数据）；方差/均值给 max_abs/max_rel 并说明为什么不能逐字节。
# 尾块 cols=1000/4097；bf16 覆盖并打印 bf16_conversions 与「改写后可疑语法 0」。
# 跑法：python tools/remote.py sbx-run probes/197_var_axis.sh --name 197_var_axis --timeout 600

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

cat > "$W/var_axis.py" <<'PYEOF'
"""row_var / row_std / sum_axis / mean_axis 验收。"""

import ctypes
import math
import struct
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_reduce as rdc


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def to_bits(v, dtype):
    if dtype == "fp32":
        return struct.unpack("<I", struct.pack("<f", v))[0]
    if dtype == "fp16":
        return struct.unpack("<H", struct.pack("<e", v))[0]
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def from_bits(b, dtype):
    if dtype == "fp32":
        return struct.unpack("<f", struct.pack("<I", b))[0]
    if dtype == "fp16":
        return struct.unpack("<e", struct.pack("<H", b))[0]
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


def ss(dtype):
    return 4 if dtype == "fp32" else 2


def fmt(dtype):
    return {"fp32": "<I", "fp16": "<H", "bf16": "<H"}[dtype]


def round_to(v, dtype):
    return from_bits(to_bits(v, dtype), dtype)


def pack(values, dtype):
    f = fmt(dtype)
    return b"".join(struct.pack(f, to_bits(v, dtype)) for v in values)


def unpack(raw, n, dtype):
    f = fmt(dtype)
    return [from_bits(struct.unpack_from(f, raw, ss(dtype) * i)[0], dtype) for i in range(n)]


def to_dev(raw):
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def read(ptr, n, dtype):
    buf = ctypes.create_string_buffer(ss(dtype) * max(n, 1))
    rt.memcpy(buf, ptr, ss(dtype) * n, rt.MEMCPY_DEVICE_TO_HOST)
    return unpack(buf.raw, n, dtype)


def round_f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


# --------------------------------------------------------------------------- #
# §1 row_var / row_std
# --------------------------------------------------------------------------- #

def variance_case(dtype):
    """构造「均值远大于标准差」的行——这是一趟公式的杀手场景。

    基准值按 dtype 的表示范围挑：fp32 用 1e6（相消最惨），fp16 最大才 65504、bf16 的 ulp 更粗，
    所以它们的「均值/标准差」比只能取到 ~1e3 量级——但判的不是相消有多惨，而是**实现是否与
    双精度参照一致**，相消场景只在 fp32 那档给出量化对照。
    """
    base = {"fp32": 1e6, "fp16": 256.0, "bf16": 64.0}[dtype]
    step = {"fp32": 0.25, "fp16": 0.25, "bf16": 0.5}[dtype]
    tol = {"fp32": 1e-3, "fp16": 5e-3, "bf16": 2e-2}[dtype]
    ok = total = 0
    print("=== §1 row_var / row_std（两趟） ===")
    for rows, cols in ((5, 1000), (7, 512), (9, 4097)):
        data, rowdata = [], []
        for r in range(rows):
            row = [round_to(base + (c % 5) * step, dtype) for c in range(cols)]
            rowdata.append(row)
            data.extend(row)
        px = to_dev(pack(data, dtype))
        po = to_dev(b"\x00" * (ss(dtype) * rows))

        for correction, tag in ((1.0, "样本(c=1)"), (0.0, "总体(c=0)")):
            # 参照：双精度两趟
            ref = []
            for row in rowdata:
                mean = sum(row) / cols
                m2 = sum((v - mean) ** 2 for v in row)
                ref.append(m2 / (cols - correction))
            rdc.row_var(px, po, rows, cols, dtype, correction)
            rt.synchronize()
            got = read(po, rows, dtype)
            ma = max(abs(g - e) for g, e in zip(got, ref))
            rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 0)

            # 对照 1：一趟公式在**双精度**下（只剩公式自己的相消误差）
            naive_d, naive_f = [], []
            for row in rowdata:
                n = float(cols)
                sumsq_d = sum(v * v for v in row)
                sum_d = sum(row)
                naive_d.append((sumsq_d - sum_d * sum_d / n) / (n - correction))
                # 对照 2：一趟公式在**fp32 顺序累加**下（真实的一趟内核会拿到的数）
                s1 = 0.0
                s2 = 0.0
                for v in row:
                    s1 = round_f32(s1 + v)
                    s2 = round_f32(s2 + v * v)
                mean_f = round_f32(s1 / n)
                naive_f.append(round_f32((s2 - round_f32(n * mean_f * mean_f)) / (n - correction)))
            err_d = max(abs(a - b) for a, b in zip(naive_d, ref))
            err_f = max(abs(a - b) for a, b in zip(naive_f, ref))
            total += 1
            good = ma <= tol
            ok += 1 if good else 0
            print(f"  row_var {dtype:4s} rows={rows} cols={cols:4d} {tag:9s} "
                  f"max_abs={ma:.3e} max_rel={rel:.3e} 限={tol:.0e} → {good}")
            if correction == 1.0 and dtype == "fp32":
                # 这个比值只在 fp32 档有意义：fp16/bf16 的输出本身就比对照少几位精度，
                # 「我们 vs fp32 一趟」在那些档是苹果比橘子（会在某些形状上出现 <1 的假象）。
                print(f"      对照：一趟公式(双精度)误差={err_d:.3e}   一趟公式(fp32顺序累加)误差"
                      f"={err_f:.3e}   → 两趟比 fp32 一趟好 "
                      f"{err_f / ma if ma else float('inf'):.3g} 倍")

        # row_std：与 sqrt(参考方差) 比
        rdc.row_std(px, po, rows, cols, dtype, 1.0)
        rt.synchronize()
        got = read(po, rows, dtype)
        ref = []
        for row in rowdata:
            mean = sum(row) / cols
            m2 = sum((v - mean) ** 2 for v in row)
            ref.append(math.sqrt(m2 / (cols - 1)))
        rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 0)
        total += 1
        good = rel <= {"fp32": 1e-5, "fp16": 1e-2, "bf16": 5e-2}[dtype]
        ok += 1 if good else 0
        print(f"  row_std {dtype:4s} rows={rows} cols={cols:4d} max_rel={rel:.3e} → {good}")
        rt.free(px)
        rt.free(po)
    print(f"§1 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §2 sum_axis / mean_axis
# --------------------------------------------------------------------------- #

def axis_case(dtype):
    ok = total = 0
    print("=== §2 sum_axis / mean_axis（单个中间维） ===")
    for outer, red, inner in ((4, 3, 1000), (3, 17, 4097), (5, 8, 512)):
        # dyadic 数据：和精确，可要求逐字节
        flat = []
        for o in range(outer):
            for r in range(red):
                for i in range(inner):
                    flat.append(round_to(((o + r + i) % 9 - 4) * 0.5, dtype))
        px = to_dev(pack(flat, dtype))
        po = to_dev(b"\x00" * (ss(dtype) * outer * inner))

        def cpu_sum():
            out = [0.0] * (outer * inner)
            for o in range(outer):
                for r in range(red):
                    base = (o * red + r) * inner
                    for i in range(inner):
                        out[o * inner + i] += flat[base + i]
            return out

        ref_sum = cpu_sum()

        rdc.sum_axis(px, po, outer, red, inner, dtype)
        rt.synchronize()
        got = read(po, outer * inner, dtype)
        # dyadic 且量级小：fp32 累加没有舍入，可以和 CPU 参照逐字节比
        expect_bits = pack([round_to(v, dtype) for v in ref_sum], dtype)
        view = b"".join(struct.pack(fmt(dtype), to_bits(v, dtype)) for v in got)
        total += 1
        byte_same = view == expect_bits
        ok += 1 if byte_same else 0
        ma = max(abs(g - e) for g, e in zip(got, ref_sum))
        print(f"  sum_axis  {dtype:4s} ({outer},{red},{inner:4d}) 逐字节相等={byte_same} "
              f"max_abs={ma:.3e}")

        rdc.mean_axis(px, po, outer, red, inner, dtype)
        rt.synchronize()
        got = read(po, outer * inner, dtype)
        ref_mean = [v / red for v in ref_sum]
        ma = max(abs(g - e) for g, e in zip(got, ref_mean))
        total += 1
        good = ma <= {"fp32": 1e-6, "fp16": 1e-3, "bf16": 5e-3}[dtype]
        ok += 1 if good else 0
        print(f"  mean_axis {dtype:4s} ({outer},{red},{inner:4d}) max_abs={ma:.3e} → {good}")
        rt.free(px)
        rt.free(po)
    print(f"§2 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §3 bf16 体检
# --------------------------------------------------------------------------- #

BAD = ("fpext", "fptrunc", "trunc nuw", "trunc nsw", "splat", "fp_extend", "fp_round")


def health():
    from corex_torch import _bf16
    print("=== §3 bf16 体检（row_var / sum_axis） ===")
    conv_total = bad_total = 0
    for key, kernel in sorted(rdc._KERNEL_CACHE.items(), key=lambda kv: str(kv[0])):
        if "bf16" not in str(key):
            continue
        if not any(s in str(key) for s in ("row_var", "sum_axis")):
            continue
        counts = getattr(kernel.metadata, "corex_rewrite_counts", None) or {}
        conv = int(counts.get("bf16_conversions", 0) or 0)
        rewritten, _ = _bf16.rewrite_bf16_conversions(kernel.asm.get("biir", ""))
        left = {b: rewritten.count(b) for b in BAD if rewritten.count(b)}
        conv_total += conv
        bad_total += sum(left.values())
        print(f"  {str(key):48s} cubin={len(kernel.asm['cubin']):5d} "
              f"bf16_conversions={conv:2d} 改写后可疑语法={left or '{}'}")
    print(f"  合计 bf16_conversions={conv_total}，改写后可疑语法={bad_total}（必须 0）")
    return conv_total, bad_total


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    ok1 = t1 = ok2 = t2 = 0
    for dtype in ("fp32", "fp16", "bf16"):
        a, b = variance_case(dtype)
        ok1 += a
        t1 += b
        c, d = axis_case(dtype)
        ok2 += c
        t2 += d
    conv, bad = health()
    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if (ok1 == t1 and ok2 == t2 and bad == 0) else "FAIL"
    print(f"§1 {ok1}/{t1}  §2 {ok2}/{t2}  bf16 改写命中 {conv}  可疑语法 {bad}")
    print("VARAXIS_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. row_var / sum_axis 验收 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/var_axis.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE197_DONE py_rc=$PYRC"
exit "$PYRC"
