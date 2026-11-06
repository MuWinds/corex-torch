set -u
# 目的（bf16 通路的**技术前提验证**，先于实现）：确认「缓冲区按 u16 重新解释 + 整数位运算自己
# 造转换」这条路在这张卡上真的成立，并且 IR 里**不会**出现厂商 llc 缺的那两条转换指令。
#
# 要回答的五件事（每件都会打印 biir 里的 fp_extend / fp_round 命中数）：
#   1. 展宽：`load u16 → uint32 → <<16 → bitcast f32`，且用 **@triton.jit 辅助函数**包装
#      （实现里要这么复用，先验证它真的被内联、不额外出转换指令）；
#   2. 收窄：`bitcast u32 → (+0x7FFF+lsb) → >>16 → u16`，是否与硬件的 RNE 结果**逐字节**一致；
#   3. 纯 bf16 算术（`*bf16` 进、`*bf16` 出、无 fp32 标量）：mul / div / copy 各自编不编得过；
#   4. `add` 带 fp32 `alpha`：是否必然引入 `fp_extend`（这决定 add/sub 必须走 fp32 + 位运算）；
#   5. 结论：每个算子该走哪条路，用它自己的 grep 数字说话。
#
# 判定：§1/§2 两条位运算路必须 OK 且 `fp_extend`/`fp_round` 命中为 0。若走不通就 FAIL 并停下来
# 报 Lead（**不退回纯 bf16 硬算**——那等于把数值不可用的内核伪装成支持 bf16）。
#
# 载荷：`work/ops/payload.tgz`。
# 跑法：python tools/remote.py sbx-run probes/190_bf16_path.sh --name 190_bf16_path --timeout 600

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

cat > "$W/bf16_path.py" <<'PYEOF'
"""bf16 位运算通路验证：能不能绕开 fp_extend / fp_round。"""

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


# --------------------------------------------------------------------------- #
# 实现里要用的两个 @triton.jit 辅助函数（先在这里验证「能被内联且不额外出转换」）
# --------------------------------------------------------------------------- #

@triton.jit
def _bf16_widen(raw):
    """u16 位模式 → f32 值。整数移位 + bitcast，不产生 fp_extend。"""
    return (raw.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_narrow(f):
    """f32 值 → u16 位模式，round-to-nearest-even（+0x7FFF+lsb），不产生 fp_round。"""
    u = f.to(tl.uint32, bitcast=True)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).to(tl.uint16)


@triton.jit
def k_widen_sum(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    f = _bf16_widen(tl.load(a_ptr + off, mask=mask, other=0))
    tl.store(out_ptr, tl.sum(f, axis=0))


@triton.jit
def k_narrow_fill(out_ptr, n, value, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, _bf16_narrow(value), mask=off < n)


@triton.jit
def k_mul_bf16(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=mask) * tl.load(b_ptr + off, mask=mask),
             mask=mask)


@triton.jit
def k_div_bf16(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=mask) / tl.load(b_ptr + off, mask=mask),
             mask=mask)


@triton.jit
def k_copy_bf16(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=mask), mask=mask)


@triton.jit
def k_add_fp32_alpha(a_ptr, b_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask)
    b = tl.load(b_ptr + off, mask=mask)
    tl.store(o_ptr + off, a + alpha * b, mask=mask)


@triton.jit
def k_add_bf16_mathtrick(a_ptr, b_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    """add(alpha) 的候选实现：fp32 里算 + 位运算展宽/收窄（不含任何 fp 转换指令）。"""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    a = _bf16_widen(tl.load(a_ptr + off, mask=mask, other=0))
    b = _bf16_widen(tl.load(b_ptr + off, mask=mask, other=0))
    tl.store(o_ptr + off, _bf16_narrow(a + alpha * b), mask=mask)


# --------------------------------------------------------------------------- #
# 主机侧工具
# --------------------------------------------------------------------------- #

def f32_to_bf16_bits(v):
    """RNE，与硬件 bf16 转换一致。"""
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_bits_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


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


def grep(kernel):
    """biir 里两条转换指令的命中数——这是「真的绕开了」的唯一硬证据。"""
    biir = kernel.asm.get("biir", "")
    return biir.count("fp_extend"), biir.count("fp_round")


def try_compile(name, fn, sig, cexpr, num_warps=4):
    try:
        k = ct.compile_kernel(fn, sig, cexpr, num_warps=num_warps)
    except Exception as exc:                            # noqa: BLE001 - 失败也是结论
        print(f"  {name:24s} 编译失败 {type(exc).__name__}: "
              f"{str(exc).replace(chr(10), ' ')[:200]}")
        return None
    ext, rnd = grep(k)
    print(f"  {name:24s} cubin={len(k.asm['cubin']):5d} B  fp_extend={ext}  fp_round={rnd}")
    return k


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    ok = fail = 0
    n, BLOCK = 1024, 1024
    vals = [0.5, -1.25, 3.0, -0.0625] * (n // 4)
    ptr_ty = "*u16"

    print("=== §1 展宽：u16 → uint32 → <<16 → bitcast f32（经 @triton.jit 辅助函数）===")
    src = to_dev(pack_bf16(vals))
    outp = rt.malloc(4)
    k1 = try_compile("k_widen_sum", k_widen_sum,
                     {"a_ptr": ptr_ty, "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
                     {"BLOCK": BLOCK})
    if k1 is None:
        fail += 1
    else:
        ct.launch(k1, (1,), src, outp, n)
        rt.synchronize()
        buf = ctypes.create_string_buffer(4)
        rt.memcpy(buf, outp, 4, rt.MEMCPY_DEVICE_TO_HOST)
        got = struct.unpack("<f", buf.raw)[0]
        ref = sum(bf16_bits_to_f32(f32_to_bf16_bits(v)) for v in vals)
        same = abs(got - ref) < 1e-6 * max(1.0, abs(ref))
        ext, rnd = grep(k1)
        good = same and ext == 0 and rnd == 0
        print(f"      got={got:.6f} ref={ref:.6f} → {same}；转换指令命中 {ext + rnd}（应 0）"
              f" → {'OK' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print("=== §2 收窄：RNE 位运算，与硬件转换逐字节比对 ===")
    dst = to_dev(b"\x00" * (n * 2))
    k2 = try_compile("k_narrow_fill", k_narrow_fill,
                     {"out_ptr": ptr_ty, "n": "i32", "value": "fp32", "BLOCK": "constexpr"},
                     {"BLOCK": BLOCK})
    if k2 is None:
        fail += 1
    else:
        for v in (0.5, -2.0, 1.0 / 3.0, 1e-40):
            ct.launch(k2, (1,), dst, n, v)
            rt.synchronize()
            bits = read_u16(dst, 4)
            exp = f32_to_bf16_bits(v)
            same = all(b == exp for b in bits)
            print(f"      fill({v!r:>10}) → 0x{bits[0]:04X}（RNE 期望 0x{exp:04X}）→ {same}")
            ok += 1 if same else 0
            fail += 0 if same else 1
        ext, rnd = grep(k2)
        good = ext == 0 and rnd == 0
        print(f"      转换指令命中 {ext + rnd}（应 0）→ {'OK' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print("=== §3 纯 bf16 算术（*bf16 进、*bf16 出、无 fp32 标量）===")
    a16 = to_dev(pack_bf16([0.5, -1.0, 2.0, -0.25] * (n // 4)))
    b16 = to_dev(pack_bf16([2.0, 0.5, -4.0, 8.0] * (n // 4)))
    o16 = to_dev(b"\x00" * (n * 2))
    for name, fn, expect, sig in (
        ("mul", k_mul_bf16, [1.0, -0.5, -8.0, -2.0],
         {"a_ptr": "*bf16", "b_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}),
        ("div", k_div_bf16, [0.25, -2.0, -0.5, -0.03125],
         {"a_ptr": "*bf16", "b_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}),
        ("copy", k_copy_bf16, [0.5, -1.0, 2.0, -0.25],
         {"a_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}),
    ):
        k = try_compile(f"k_{name}_bf16", fn, sig, {"BLOCK": BLOCK})
        if k is None:
            fail += 1
            continue
        if name == "copy":
            ct.launch(k, (1,), a16, o16, n)
        else:
            ct.launch(k, (1,), a16, b16, o16, n)
        rt.synchronize()
        got = [bf16_bits_to_f32(b) for b in read_u16(o16, 4)]
        same = all(abs(g - e) < 1e-9 for g, e in zip(got, expect))
        exp_bits = [f32_to_bf16_bits(e) for e in expect]
        got_bits = read_u16(o16, 4)
        byte_same = got_bits == exp_bits
        ext, rnd = grep(k)
        good = same and byte_same and ext == 0 and rnd == 0
        print(f"      {name}: got={got} 期望={expect} 逐字节={byte_same}；"
              f"转换指令 {ext + rnd}（应 0）→ {'纯 bf16 可用' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print("=== §4 add 带 fp32 alpha：两条路对比 ===")
    sig_alpha = {"a_ptr": "*bf16", "b_ptr": "*bf16", "o_ptr": "*bf16", "n": "i32",
                 "alpha": "fp32", "BLOCK": "constexpr"}
    k4 = try_compile("k_add_fp32_alpha", k_add_fp32_alpha, sig_alpha, {"BLOCK": BLOCK})
    if k4 is None:
        print("      → 果然编不过（或编过但含 fp_extend）：add/sub 只能走 fp32 + 位运算")
    else:
        ext, rnd = grep(k4)
        print(f"      → 编过了，但 fp_extend={ext} fp_round={rnd}"
              f"（>0 就说明它偷偷插了转换）")
    k5 = try_compile("k_add_bf16_mathtrick", k_add_bf16_mathtrick, sig_alpha, {"BLOCK": BLOCK})
    if k5 is None:
        fail += 1
    else:
        ct.launch(k5, (1,), a16, b16, o16, n, 1.0)
        rt.synchronize()
        got = [bf16_bits_to_f32(b) for b in read_u16(o16, 4)]
        exp = [0.5 + 2.0, -1.0 + 0.5, 2.0 + -4.0, -0.25 + 8.0]
        byte_same = read_u16(o16, 4) == [f32_to_bf16_bits(e) for e in exp]
        ext, rnd = grep(k5)
        good = byte_same and ext == 0 and rnd == 0
        print(f"      mathtrick: got={got} 期望={exp} 逐字节={byte_same}；"
              f"转换指令 {ext + rnd}（应 0）→ {'可用' if good else 'FAIL'}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print("[end] torch_modules =", len(torchmods()))
    print(f"通过 {ok}，失败 {fail}")
    print("BF16PATH_VERDICT", "PASS" if fail == 0 else "FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. bf16 位运算通路验证 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/bf16_path.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE190_DONE py_rc=$PYRC"
exit "$PYRC"
