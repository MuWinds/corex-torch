set -u
# 目的：第 5 批算子的真机验收（Lead 指定的优先级顺序）。
#
#   1. relu / relu_        —— 逐字节相等（max 没有舍入），原地安全
#   2. rsqrt               —— max_rel（rsqrt.approx 是近似指令），并与 1/tl.sqrt 对照
#   3. gelu                —— erf 精确版 + tanh 近似版，**各自对各自的数学形式**取参照
#   4. masked_fill         —— 逐字节相等；mask 是 bool（1 字节/元素）
#   5. max.dim / argmax    —— 值逐字节相等；下标按 aten 语义**取第一个**最大值（重值用例）
#   6. _log_softmax.out    —— 与现有 softmax 共用一条内核（LOG constexpr），不写第二个内核
#
# 每个算子都覆盖 fp32/fp16/bf16 × n=4096/4097/1000（尾块非 2 的幂、非整块）。
# bf16 一律给 `bf16_conversions` 命中数与「改写后可疑语法 = 0」的体检行。
#
# 跑法：python tools/remote.py sbx-run probes/196_new_ops.sh --name 196_new_ops --timeout 600

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

cat > "$W/new_ops.py" <<'PYEOF'
"""第 5 批算子验收：relu / rsqrt / gelu / masked_fill / max.dim+argmax / log_softmax。"""

import ctypes
import math
import struct
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _ops_elementwise as elt
from corex_torch import _ops_reduce as rdc


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


# --------------------------------------------------------------------------- #
# dtype 打包/解包（fp32 / fp16 / bf16）
# --------------------------------------------------------------------------- #

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


def nbytes(dtype):
    return 4 if dtype == "fp32" else 2


def fmt(dtype):
    return {"fp32": "<I", "fp16": "<H", "bf16": "<H"}[dtype]


def pack(values, dtype):
    f = fmt(dtype)
    return b"".join(struct.pack(f, to_bits(v, dtype)) for v in values)


def unpack(raw, n, dtype):
    f = fmt(dtype)
    return [from_bits(struct.unpack_from(f, raw, nbytes(dtype) * i)[0], dtype)
            for i in range(n)]


def to_dev(raw):
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def read(ptr, n, dtype):
    ss = nbytes(dtype)
    buf = ctypes.create_string_buffer(ss * max(n, 1))
    rt.memcpy(buf, ptr, ss * n, rt.MEMCPY_DEVICE_TO_HOST)
    return buf.raw


def round_to(v, dtype):
    """把双精度值按 dtype 舍入一次（bf16/fp16 用 RNE，与硬件一致）。"""
    return from_bits(to_bits(v, dtype), dtype)


def read_i64(ptr, n):
    buf = ctypes.create_string_buffer(8 * max(n, 1))
    rt.memcpy(buf, ptr, 8 * n, rt.MEMCPY_DEVICE_TO_HOST)
    return [struct.unpack_from("<q", buf.raw, 8 * i)[0] for i in range(n)]


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def data_in(n, dtype):
    """覆盖负值/正值/零附近/大值，且都能在 fp16/bf16 里精确表示或至少有限。"""
    out = []
    for i in range(n):
        k = i % 8
        if k == 0:
            v = 0.5
        elif k == 1:
            v = -1.25
        elif k == 2:
            v = 3.0
        elif k == 3:
            v = -0.0625
        elif k == 4:
            v = 7.5
        elif k == 5:
            v = -4.0
        elif k == 6:
            v = 0.25
        else:
            v = -8.0
        out.append(round_to(v * (1 + (i // 8) % 3 * 0.5), dtype))
    return out


def data_pos(n, dtype):
    """rsqrt 的输入必须为正：用 2 的幂与 0.5 的倍数（精确可表示）。"""
    out = []
    for i in range(n):
        k = i % 6
        v = [0.25, 0.5, 1.0, 2.0, 4.0, 16.0][k]
        out.append(round_to(v * (1 + (i // 6) % 2), dtype))
    return out


# --------------------------------------------------------------------------- #
# §1 一元 / 选择类
# --------------------------------------------------------------------------- #

def unary_and_select():
    print("=== §1 relu / relu_ / rsqrt / gelu / masked_fill ===")
    ok = total = 0
    for dtype in ("fp32", "fp16", "bf16"):
        for n in (4096, 4097, 1000):
            xs = data_in(n, dtype)
            px = to_dev(pack(xs, dtype))
            po = to_dev(b"\x00" * (nbytes(dtype) * n))

            def cmp_exact(name, call, expect):
                nonlocal ok, total
                call()
                rt.synchronize()
                got = read(po, n, dtype)
                exp = pack(expect, dtype)
                total += 1
                good = got == exp
                ok += 1 if good else 0
                extra = "" if good else " 首个不同=" + str(
                    next((i for i in range(n)
                          if got[nbytes(dtype) * i:nbytes(dtype) * (i + 1)]
                          != exp[nbytes(dtype) * i:nbytes(dtype) * (i + 1)]), -1))
                print(f"  {name:22s} {dtype:4s} n={n:5d} 逐字节相等={good}{extra}")

            def cmp_tol(name, call, expect_double, tol, note=""):
                nonlocal ok, total
                call()
                rt.synchronize()
                got = unpack(read(po, n, dtype), n, dtype)
                # 参照也按 dtype 舍入一次：实现误差里必然含一次输出舍入
                exp = [round_to(e, dtype) for e in expect_double]
                ma = max(abs(g - e) for g, e in zip(got, exp))
                # 相对误差只在「量级与最大值相当」的元素上取：GELU 在 x 很负时结果 ~0
                # （-8*Φ(-8) ≈ 5e-15，fp32 下直接下溢成 0），拿它做分母得到的 1.0 是伪影，
                # 不是实现误差。这一条对 gelu 尤其重要：不筛就会报 max_rel=1.0。
                scale = max(abs(e) for e in exp) or 1.0
                rel = max((abs(g - e) / abs(e)) for g, e in zip(got, exp)
                          if abs(e) > 1e-3 * scale)
                total += 1
                good = ma <= tol
                ok += 1 if good else 0
                print(f"  {name:22s} {dtype:4s} n={n:5d} max_abs={ma:.3e} max_rel={rel:.3e}"
                      f" 限={tol:.1e} → {good}{note}")

            # --- relu：逐字节相等 ---
            cmp_exact("relu", lambda: elt.relu(px, po, n, dtype),
                      [max(v, 0.0) for v in xs])

            # --- relu_：原地（out==in），结果与 relu 相同，且输入被真的改写 ---
            pin = to_dev(pack(xs, dtype))
            elt.relu(pin, pin, n, dtype)
            rt.synchronize()
            got = read(pin, n, dtype)
            total += 1
            same = got == pack([max(v, 0.0) for v in xs], dtype)
            ok += 1 if same else 0
            print(f"  {'relu_ 原地':22s} {dtype:4s} n={n:5d} 逐字节相等={same}")
            rt.free(pin)

            # --- rsqrt：sqrt.approx + div.rn，约 2 ulp，按 max_rel 判 ---
            ys = data_pos(n, dtype)
            pr = to_dev(pack(ys, dtype))
            elt.rsqrt(pr, po, n, dtype)
            rt.synchronize()
            got = unpack(read(po, n, dtype), n, dtype)
            exp = [round_to(1.0 / math.sqrt(v), dtype) for v in ys]
            ma = max(abs(g - e) for g, e in zip(got, exp))
            rel = max((abs(g - e) / abs(e)) for g, e in zip(got, exp) if abs(e) > 1e-30)
            tol = {"fp32": 2e-6, "fp16": 4e-3, "bf16": 2e-2}[dtype]
            total += 1
            good = rel <= tol
            ok += 1 if good else 0
            print(f"  {'rsqrt':22s} {dtype:4s} n={n:5d} max_abs={ma:.3e} max_rel={rel:.3e}"
                  f" 限={tol:.0e} → {good}")
            rt.free(pr)

            # --- gelu：erf 精确版 ---
            cmp_tol("gelu(erf)",
                    lambda: elt.gelu(px, po, n, dtype, "erf"),
                    [0.5 * v * (1 + math.erf(v / math.sqrt(2.0))) for v in xs],
                    {"fp32": 2e-6, "fp16": 4e-3, "bf16": 2e-2}[dtype])

            # --- gelu：tanh 近似版（参照用同一条公式，测实现不测近似） ---
            def gelu_tanh(v):
                return 0.5 * v * (1 + math.tanh(math.sqrt(2.0 / math.pi)
                                                * (v + 0.044715 * v ** 3)))
            cmp_tol("gelu(tanh)",
                    lambda: elt.gelu(px, po, n, dtype, "tanh"),
                    [gelu_tanh(v) for v in xs],
                    {"fp32": 2e-6, "fp16": 4e-3, "bf16": 2e-2}[dtype])
            # 两种形式的差本身有多大（近似误差，不是实现误差）：只在 fp32 下报一次
            if dtype == "fp32" and n == 4097:
                d = max(abs(gelu_tanh(v) - 0.5 * v * (1 + math.erf(v / math.sqrt(2.0))))
                        for v in xs)
                print(f"  {'[参考] tanh近似误差':22s} fp32 n={n} max_abs={d:.3e}"
                      f"（这是两种数学形式的差，与实现无关）")

            # --- masked_fill：逐字节相等，含原地 ---
            mv = [1 if (i % 3 == 0) else 0 for i in range(n)]
            pm = to_dev(bytes(mv))
            cmp_exact("masked_fill(0.5)",
                      lambda: elt.masked_fill(px, pm, po, n, dtype, 0.5),
                      [0.5 if mv[i] else xs[i] for i in range(n)])
            # masked_fill_ 原地
            pin = to_dev(pack(xs, dtype))
            elt.masked_fill(pin, pm, pin, n, dtype, -2.0)
            rt.synchronize()
            got = read(pin, n, dtype)
            total += 1
            same = got == pack([-2.0 if mv[i] else xs[i] for i in range(n)], dtype)
            ok += 1 if same else 0
            print(f"  {'masked_fill_ 原地':22s} {dtype:4s} n={n:5d} 逐字节相等={same}")
            rt.free(pin)
            rt.free(pm)
            rt.free(px)
            rt.free(po)
    print(f"§1 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §2 max.dim / argmax / log_softmax
# --------------------------------------------------------------------------- #

def reductions():
    print("=== §2 max.dim / argmax / _log_softmax ===")
    ok = total = 0
    for dtype in ("fp32", "fp16", "bf16"):
        for rows, cols in ((5, 1000), (7, 512), (9, 4097)):
            flat, rowsdata = [], []
            for r in range(rows):
                row = [round_to(((r * 13 + c * 7) % 41 - 20) * 0.5, dtype) for c in range(cols)]
                # 制造重复最大值，专门验「取第一个」的语义
                if cols > 4:
                    row[cols - 1] = row[1]
                    row[0] = row[1]
                rowsdata.append(row)
                flat.extend(row)
            px = to_dev(pack(flat, dtype))
            pval = to_dev(b"\x00" * (nbytes(dtype) * rows))
            pidx = rt.malloc(8 * rows)

            # --- max.dim：值与下标 ---
            rdc.max_dim(px, pval, pidx, rows, cols, dtype)
            rt.synchronize()
            got_v = read(pval, rows, dtype)
            exp_v = pack([max(r) for r in rowsdata], dtype)
            got_i = read_i64(pidx, rows)
            exp_i = [r.index(max(r)) for r in rowsdata]      # list.index = 第一个
            total += 1
            good = got_v == exp_v and got_i == exp_i
            ok += 1 if good else 0
            print(f"  max.dim  {dtype:4s} rows={rows} cols={cols:4d} 值逐字节={got_v == exp_v}"
                  f" 下标全对={got_i == exp_i}（第一个最大值）→ {good}")
            if not (got_v == exp_v and got_i == exp_i):
                print(f"      下标 got={got_i[:6]} exp={exp_i[:6]}")

            # --- argmax：只出下标 ---
            rdc.argmax_last(px, pidx, rows, cols, dtype)
            rt.synchronize()
            got_i2 = read_i64(pidx, rows)
            total += 1
            ok += 1 if got_i2 == exp_i else 0
            print(f"  argmax   {dtype:4s} rows={rows} cols={cols:4d} 下标全对={got_i2 == exp_i}")

            # --- log_softmax：max_abs/max_rel + exp 后行和 ≈ 1 ---
            pout = to_dev(b"\x00" * (nbytes(dtype) * rows * cols))
            rdc.log_softmax_last(px, pout, rows, cols, dtype)
            rt.synchronize()
            got = unpack(read(pout, rows * cols, dtype), rows * cols, dtype)
            ref = []
            for row in rowsdata:
                m = max(row)
                ls = m + math.log(sum(math.exp(v - m) for v in row))
                ref.extend([v - ls for v in row])
            ma = max(abs(g - e) for g, e in zip(got, ref))
            rel = max((abs(g - e) / abs(e)) for g, e in zip(got, ref) if abs(e) > 1e-3)
            # exp(log_softmax) 的行和应当 ≈ 1（这是 log-softmax 最实用的自检）
            rowsum = max(abs(sum(math.exp(got[r * cols + c]) for c in range(cols)) - 1.0)
                         for r in range(rows))
            tol_a = {"fp32": 1e-4, "fp16": 2e-2, "bf16": 6e-2}[dtype]
            total += 1
            good = ma <= tol_a
            ok += 1 if good else 0
            print(f"  log_softmax {dtype:4s} rows={rows} cols={cols:4d} max_abs={ma:.3e}"
                  f" max_rel={rel:.3e} exp后行和偏离1={rowsum:.3e} 限={tol_a:.0e} → {good}")

            # --- 一致性：softmax == exp(log_softmax) ---
            psm = to_dev(b"\x00" * (nbytes(dtype) * rows * cols))
            rdc.softmax_last(px, psm, rows, cols, dtype)
            rt.synchronize()
            sm = unpack(read(psm, rows * cols, dtype), rows * cols, dtype)
            dif = max(abs(math.exp(got[i]) - sm[i]) for i in range(rows * cols))
            total += 1
            ok += 1 if dif <= tol_a else 0
            print(f"  {'一致性 exp(logsm)==sm':26s} rows={rows} cols={cols:4d} max_abs={dif:.3e}"
                  f" → {dif <= tol_a}")

            for p in (px, pval, pidx, pout, psm):
                rt.free(p)
    print(f"§2 {ok}/{total}")
    return ok, total


# --------------------------------------------------------------------------- #
# §3 bf16 体检
# --------------------------------------------------------------------------- #

BAD = ("fpext", "fptrunc", "trunc nuw", "trunc nsw", "splat", "fp_extend", "fp_round")


def health():
    from corex_torch import _bf16
    print("=== §3 bf16 体检（本轮新增内核）===")
    total_conv = 0
    bad = 0
    for label, cache in (("elementwise", elt._KERNEL_CACHE), ("reduce", rdc._KERNEL_CACHE)):
        rows = [k for k in cache
                if "bf16" in str(k) and any(s in str(k) for s in
                                            ("relu", "rsqrt", "gelu", "masked_fill",
                                             "softmax", "max_dim"))]
        print(f"--- {label}：{len(rows)} 个新 bf16 内核 ---")
        for key in sorted(map(str, rows)):
            pass
        for key, kernel in sorted(cache.items(), key=lambda kv: str(kv[0])):
            if key not in rows:
                continue
            md = kernel.metadata
            counts = getattr(md, "corex_rewrite_counts", None) or {}
            conv = int(counts.get("bf16_conversions", 0) or 0)
            rewritten, _ = _bf16.rewrite_bf16_conversions(kernel.asm.get("biir", ""))
            left = {b: rewritten.count(b) for b in BAD if rewritten.count(b)}
            total_conv += conv
            bad += sum(left.values())
            print(f"  {str(key):52s} cubin={len(kernel.asm['cubin']):5d} "
                  f"bf16_conversions={conv:2d} 改写后可疑语法={left or '{}'}")
    print(f"  新内核 bf16_conversions 合计={total_conv}，改写后可疑语法={bad}（必须 0）")
    return total_conv, bad


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    ok1, t1 = unary_and_select()
    ok2, t2 = reductions()
    conv, bad = health()
    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if (ok1 == t1 and ok2 == t2 and bad == 0 and conv > 0) else "FAIL"
    print(f"§1 {ok1}/{t1}  §2 {ok2}/{t2}  bf16 改写命中 {conv}  可疑语法 {bad}")
    print("NEWOPS_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 第 5 批算子验收 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/new_ops.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE196_DONE py_rc=$PYRC"
exit "$PYRC"
