set -u
# 目的：`probes/190` 证实「展宽」方向通了（`fp_extend=0`），但「收窄」方向撞在
# `trunc nuw i32 → i16`（LLVM 17 起 trunc 才有 nuw/nsw 标记，厂商 llc 是 16，直接 parse error）。
# 本探针只问一件事：**有没有不改 corex-triton 就能收窄的办法**，顺便把纯 bf16 算术那批
# （mul / copy / gather / add-sub 无 alpha）的逐字节正确性钉下来。
#
# 三条候选收窄路：
#   A. `.to(tl.uint16)`（基准）：预期 `trunc nuw` → llc 拒绝；
#   B. `tl.view(u32_vec, tl.uint16)` 把 u32 向量按位宽劈成两半：如果 lowering 成 bitcast
#      而不是 trunc，就不带 nuw 标记，能过；
#   C. u32 **成对打包**：不再逐元素存 u16，而是把相邻两个 bf16 打进一个 u32 存 `*u32`
#      （需要 n 为偶数；奇数尾巴单独报，不偷偷越界）。这条路一定避开 trunc，但内核要按
#      对处理。
#
# 判定：A 必须复现失败（证明问题真实）；B 或 C 至少一条通过 → 有不需要改上游的收窄路。

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

cat > "$W/bf16_narrow.py" <<'PYEOF'
"""收窄的三条候选路 + 纯 bf16 算术的逐字节正确性。"""

import ctypes
import os
import struct
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _triton as ct
import triton
import triton.language as tl


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


@triton.jit
def _bf16_bits_from_f32(f):
    """f32 → 落在低 16 位的 bf16 位模式（RNE）。**故意不 to(uint16)**，避免 trunc。"""
    u = f.to(tl.uint32, bitcast=True)
    lsb = (u >> 16) & 1
    return (u + 0x7FFF + lsb) >> 16


@triton.jit
def _bf16_widen(raw):
    """u16 位模式（u32 承载）→ f32。整数移位 + bitcast。"""
    return (raw.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


# --- A：逐元素 .to(uint16)（预期 llc 拒绝：trunc nuw）---
@triton.jit
def k_narrow_to_u16(out_ptr, n, value, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, _bf16_bits_from_f32(value).to(tl.uint16), mask=off < n)


# --- B：tl.view 把 u32 向量劈成 u16 ---
@triton.jit
def k_narrow_view(out_ptr, n, value, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK // 2)
    b = _bf16_bits_from_f32(value)                     # u32 标量，低 16 位有效
    v32 = tl.full((BLOCK // 2,), 0, tl.uint32) + b      # 广播成向量
    v32 = v32 | (v32 << 16)                             # 高低两半都填同一个值（只为验语法）
    v16 = tl.view(v32, tl.uint16)                       # <BLOCK/2 x u32> → <BLOCK x u16>
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, v16, mask=off < n)


# --- C：成对打包成 u32 存（一定避开 trunc）---
@triton.jit
def k_fill_pair(out_u32, n_pairs, value, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = _bf16_bits_from_f32(value)
    packed = b | (b << 16)
    tl.store(out_u32 + lane, packed, mask=lane < n_pairs)


@triton.jit
def k_add_pair(a_u32, b_u32, o_u32, n_pairs, alpha, BLOCK: tl.constexpr):
    """成对打包的 add：一次读两个元素（u32），fp32 算完再打包回去。"""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < n_pairs
    ra = tl.load(a_u32 + lane, mask=mask, other=0)
    rb = tl.load(b_u32 + lane, mask=mask, other=0)
    lo = _bf16_widen(ra & 0xFFFF)
    hi = _bf16_widen((ra >> 16) & 0xFFFF)
    blo = _bf16_widen(rb & 0xFFFF)
    bhi = _bf16_widen((rb >> 16) & 0xFFFF)
    o_lo = _bf16_bits_from_f32(lo + alpha * blo)
    o_hi = _bf16_bits_from_f32(hi + alpha * bhi)
    tl.store(o_u32 + lane, o_lo | (o_hi << 16), mask=mask)


# --- 纯 bf16 算术（*bf16 指针，无 fp32 标量）---
@triton.jit
def k_add_bf16(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=mask) + tl.load(b_ptr + off, mask=mask),
             mask=mask)


@triton.jit
def k_mul_bf16(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=mask) * tl.load(b_ptr + off, mask=mask),
             mask=mask)


@triton.jit
def k_gather_bf16(src_ptr, idx_ptr, o_ptr, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    src_row = tl.load(idx_ptr + row).to(tl.int64)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    mask = off < cols
    v = tl.load(src_ptr + src_row * cols.to(tl.int64) + off.to(tl.int64), mask=mask)
    tl.store(o_ptr + row.to(tl.int64) * cols.to(tl.int64) + off.to(tl.int64), v, mask=mask)


# --------------------------------------------------------------------------- #
# 主机侧
# --------------------------------------------------------------------------- #

def f32_to_bf16_bits(v):
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def pack_bf16(values):
    return b"".join(struct.pack("<H", f32_to_bf16_bits(v)) for v in values)


def to_dev(raw):
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def read_u16(ptr, k):
    buf = ctypes.create_string_buffer(2 * k)
    rt.memcpy(buf, ptr, 2 * k, rt.MEMCPY_DEVICE_TO_HOST)
    return [struct.unpack_from("<H", buf.raw, 2 * i)[0] for i in range(k)]


def read_u32(ptr, k):
    buf = ctypes.create_string_buffer(4 * k)
    rt.memcpy(buf, ptr, 4 * k, rt.MEMCPY_DEVICE_TO_HOST)
    return [struct.unpack_from("<I", buf.raw, 4 * i)[0] for i in range(k)]


def grep(kernel):
    biir = kernel.asm.get("biir", "")
    return (biir.count("fp_extend"), biir.count("fp_round"),
            biir.count("trunc nuw"), biir.count("trunc nsw"))


def try_compile(name, fn, sig, cexpr):
    try:
        k = ct.compile_kernel(fn, sig, cexpr, num_warps=4)
    except Exception as exc:                            # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        print(f"  {name:20s} 编译失败: {msg[:230]}")
        return None, msg
    e, r, tn, ts = grep(k)
    print(f"  {name:20s} OK cubin={len(k.asm['cubin']):5d} B  fp_extend={e} fp_round={r} "
          f"trunc_nuw={tn} trunc_nsw={ts}")
    return k, ""


def main():
    print("[start] torch_modules =", len(torchmods()))
    ok = fail = 0
    n, BLOCK = 1024, 1024
    dst = to_dev(b"\x00" * (n * 2))

    print("=== §1 收窄 A：逐元素 .to(uint16)（基准，预期失败）===")
    kA, msgA = try_compile("k_narrow_to_u16", k_narrow_to_u16,
                           {"out_ptr": "*u16", "n": "i32", "value": "fp32", "BLOCK": "constexpr"},
                           {"BLOCK": BLOCK})
    a_ok = kA is not None
    print(f"      A 可用={a_ok}" + ("" if a_ok else "（预期：llc 报 trunc nuw 的 parse error）"))

    print("=== §2 收窄 B：tl.view(u32 → u16) ===")
    kB, msgB = try_compile("k_narrow_view", k_narrow_view,
                           {"out_ptr": "*u16", "n": "i32", "value": "fp32", "BLOCK": "constexpr"},
                           {"BLOCK": BLOCK})
    b_ok = kB is not None
    if b_ok:
        for v in (0.5, -2.0, 1.0 / 3.0):
            ct.launch(kB, (1,), dst, n, v)
            rt.synchronize()
            bits = read_u16(dst, 4)
            exp = f32_to_bf16_bits(v)
            same = all(b == exp for b in bits)
            print(f"      fill({v!r:>10}) → 0x{bits[0]:04X}（期望 0x{exp:04X}）→ {same}")
            ok += 1 if same else 0
            fail += 0 if same else 1
    else:
        fail += 1

    print("=== §3 收窄 C：成对打包成 u32（一定避开 trunc）===")
    kC, _ = try_compile("k_fill_pair", k_fill_pair,
                        {"out_u32": "*u32", "n_pairs": "i32", "value": "fp32",
                         "BLOCK": "constexpr"}, {"BLOCK": BLOCK})
    if kC is None:
        fail += 1
        c_ok = False
    else:
        c_ok = True
        ct.launch(kC, (1,), dst, n // 2, 0.5)
        rt.synchronize()
        bits = read_u16(dst, 6)
        exp = f32_to_bf16_bits(0.5)
        same = all(b == exp for b in bits)
        print(f"      成对 fill(0.5) → {[hex(b) for b in bits[:4]]}（期望全 0x{exp:04X}）→ {same}")
        ok += 1 if same else 0
        fail += 0 if same else 1

    print("=== §4 成对 add(alpha) 端到端（fp32 算 + 位运算转换）===")
    a_host = [0.5, -1.0, 2.0, -0.25] * (n // 4)
    b_host = [2.0, 0.5, -4.0, 8.0] * (n // 4)
    pa = to_dev(pack_bf16(a_host))
    pb = to_dev(pack_bf16(b_host))
    po = to_dev(b"\x00" * (n * 2))
    kD, _ = try_compile("k_add_pair", k_add_pair,
                        {"a_u32": "*u32", "b_u32": "*u32", "o_u32": "*u32",
                         "n_pairs": "i32", "alpha": "fp32", "BLOCK": "constexpr"},
                        {"BLOCK": BLOCK})
    if kD is None:
        fail += 1
        d_ok = False
    else:
        d_ok = True
        for alpha in (1.0, 0.5):
            ct.launch(kD, (1,), pa, pb, po, n // 2, alpha)
            rt.synchronize()
            got = read_u16(po, 8)
            exp = [f32_to_bf16_bits(x + alpha * y) for x, y in zip(a_host[:8], b_host[:8])]
            same = got == exp
            print(f"      add(alpha={alpha}) 逐字节={same}  got={[hex(b) for b in got[:4]]} "
                  f"exp={[hex(b) for b in exp[:4]]}")
            ok += 1 if same else 0
            fail += 0 if same else 1

    print("=== §5 纯 bf16 算术（add/mul/gather，逐字节）===")
    o16 = to_dev(b"\x00" * (n * 2))
    for name, fn, sig, expect in (
        ("add", k_add_bf16,
         {"a_ptr": "*bf16", "b_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"},
         [0.5 + 2.0, -1.0 + 0.5, 2.0 + -4.0, -0.25 + 8.0]),
        ("mul", k_mul_bf16,
         {"a_ptr": "*bf16", "b_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"},
         [0.5 * 2.0, -1.0 * 0.5, 2.0 * -4.0, -0.25 * 8.0]),
    ):
        k, _ = try_compile(f"k_{name}_bf16", fn, sig, {"BLOCK": BLOCK})
        if k is None:
            fail += 1
            continue
        ct.launch(k, (1,), pa, pb, o16, n)
        rt.synchronize()
        got = read_u16(o16, 8)
        exp = [f32_to_bf16_bits(e) for e in expect]
        same = got[:len(exp)] == exp
        e, r, tn, ts = grep(k)
        good = same and e == 0 and r == 0
        print(f"      {name}: 逐字节={same} 转换指令={e + r} → {'纯 bf16 可用' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    # gather（纯搬运）
    rows, cols, table_rows = 4, 64, 8
    table = [float(i) * 0.5 for i in range(table_rows * cols)]
    src = to_dev(pack_bf16(table))
    idx_list = [3, 0, 7, 1]
    pidx = rt.malloc(rows * 4)
    rt.memcpy(pidx, (ctypes.c_int32 * rows)(*idx_list), rows * 4, rt.MEMCPY_HOST_TO_DEVICE)
    pg = to_dev(b"\x00" * (rows * cols * 2))
    kg, _ = try_compile("k_gather_bf16", k_gather_bf16,
                        {"src_ptr": "*bf16", "idx_ptr": "*i32", "o_ptr": "*bf16",
                         "rows": "i32", "cols": "i32", "BLOCK": "constexpr"}, {"BLOCK": 64})
    if kg is None:
        fail += 1
    else:
        ct.launch(kg, (rows, 1), src, pidx, pg, rows, cols)
        rt.synchronize()
        exp = []
        for r in idx_list:
            exp.extend(table[r * cols:(r + 1) * cols])
        got = read_u16(pg, rows * cols)
        same = got == [f32_to_bf16_bits(v) for v in exp]
        e, r, tn, ts = grep(kg)
        good = same and e == 0 and r == 0
        print(f"      gather: 逐字节={same} 转换指令={e + r} → {'纯搬运可用' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print("[end] torch_modules =", len(torchmods()))
    print(f"通过 {ok}，失败 {fail}")
    print(f"收窄三条路：A(.to u16)={a_ok}  B(tl.view)={b_ok}  C(成对打包)={c_ok}  "
          f"成对add={d_ok}")
    print("BF16NARROW_VERDICT", "PASS" if (b_ok or c_ok) else "FAIL")
    return 0 if (b_ok or c_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 收窄三条候选路 + 纯 bf16 算术 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/bf16_narrow.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE191_DONE py_rc=$PYRC"
exit "$PYRC"
