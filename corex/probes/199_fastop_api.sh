set -u
# 目的：给 C++ 快路径的两个稳定入口做**验收**（不是只打印，而是"照表调一次"证明表是对的）。
#
# 交付给 Lead 的两样：
#   1. `_ops_elementwise.get_kernel(op, dtype, variant)` → CompiledKernel（`asm["cubin"]` 直接喂
#      `shim.load_kernel`）；另有 `get_cubin()` / `get_launch_meta()` 两个便捷写法。
#   2. `_ops_elementwise.fast_op(op, dtype, variant)` → 机器可读的**实参顺序表**
#      （`args` 有序列表 + `n_ptrs/n_i32/n_f32/has_alpha/alpha_effective` + block/warps/grid）。
#
# 本探针的核心是 §2「shim 模拟器」：**只用那张表**去填指针/i32/f32 三个并行数组并按序 launch，
# 然后与直调逐个 dtype 比对。表若错位（比如 n 和 alpha 互换），结果必然不同——这是唯一能抓住
# 「顺序错了但跑完了」的办法。C++ 侧要做的正是这件事，所以这里先当一次 C++。
#
# 跑法：python tools/remote.py sbx-run probes/199_fastop_api.sh --name 199_fastop --timeout 600

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

cat > "$W/fastop_api.py" <<'PYEOF'
"""get_kernel / fast_op 的验收 + shim 模拟器。"""

import struct
import sys

import corex_torch
from corex_torch import _runtime as rt
from corex_torch import _triton as ct
from corex_torch import _ops_elementwise as elt


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def nbytes(dtype):
    return 4 if dtype == "fp32" else 2


def fmt(dtype):
    return {"fp32": "<I", "fp16": "<H", "bf16": "<H"}[dtype]


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


def pack(values, dtype):
    f = fmt(dtype)
    return b"".join(struct.pack(f, to_bits(v, dtype)) for v in values)


def to_dev(raw):
    ptr = rt.malloc(max(len(raw), 4))
    rt.memcpy(ptr, raw, len(raw), rt.MEMCPY_HOST_TO_DEVICE)
    return ptr


def read(ptr, n, dtype):
    buf = ctypes_buf(nbytes(dtype) * max(n, 1))
    rt.memcpy(buf, ptr, nbytes(dtype) * n, rt.MEMCPY_DEVICE_TO_HOST)
    return buf.raw


def ctypes_buf(n):
    import ctypes
    return ctypes.create_string_buffer(n)


# --------------------------------------------------------------------------- #
# shim 模拟器：**只读 fast_op() 的表**，按表把实参塞进三个并行数组再 launch
# --------------------------------------------------------------------------- #

def shim_launch(op, dtype, variant, *, n, alpha=1.0, scalar=0.0, value=0.0,
                ptrs, mask_ptr=None):
    """模拟 C++ shim：不行使任何内部知识，全部依赖 fast_op() 的返回值。

    Args:
        ptrs: 角色名 → 设备地址的映射（``a``/``b``/``out``/``src``/``dst``…）。
    """
    recipe = elt.fast_op(op, dtype, variant)          # ← 机器可读的表
    kernel = elt.get_kernel(op, dtype, variant)       # ← 稳定入口拿 CompiledKernel

    i32_vals = {"n": int(n)}
    f32_vals = {"alpha": float(alpha), "scalar": float(scalar), "value": float(value)}
    ptr_vals = dict(ptrs)
    if mask_ptr is not None:
        ptr_vals["mask"] = mask_ptr

    call = []
    for role, kind in recipe["args"]:
        if kind == "ptr":
            call.append(ptr_vals[role])
        elif kind == "i32":
            call.append(i32_vals[role])
        else:
            call.append(f32_vals[role])
    assert len(call) == recipe["n_ptrs"] + recipe["n_i32"] + recipe["n_f32"], \
        "表的 args 长度与三类计数不一致"
    ct.launch(kernel, recipe["grid_fn"](n), *call)
    return recipe


def main():
    print("[start] torch_modules =", len(torchmods()))
    print("device =", rt.device_name(0))
    ok = total = 0

    print("=== §1 登记表全览（fast_variants × fast_op）===")
    variants = elt.fast_variants()
    print(f"  已登记 {len(variants)} 个 (op, variant)；FASTPATH_OPS={elt.FASTPATH_OPS}")
    for (op, variant) in variants:
        for dtype in ("fp32", "fp16", "bf16"):
            r = elt.fast_op(op, dtype, variant)
            print(f"  {op:11s} {variant:6s} {dtype:4s}  kind={r['kernel_kind']:11s} "
                  f"ptr={r['n_ptrs']} i32={r['n_i32']} f32={r['n_f32']} "
                  f"has_alpha={str(r['has_alpha']):5s} alpha_eff={str(r['alpha_effective']):5s} "
                  f"block={r['block']} warps={r['num_warps']}")
            print(f"      args 顺序 = {r['args']}")
            print(f"      ptr_roles={r['ptr_roles']}  ptr_types={r['ptr_types']}")

    print("=== §2 shim 模拟器：只按表 launch，与直调逐字节比对 ===")
    n = 4097                                  # 尾块：非 2 的幂、非 BLOCK 整数倍
    a_vals = [((i % 17) - 8) * 0.25 for i in range(n)]
    b_vals = [float([0.5, 1.0, 2.0, 4.0][i % 4]) for i in range(n)]
    for dtype in ("fp32", "fp16", "bf16"):
        a16 = [from_bits(to_bits(v, dtype), dtype) for v in a_vals]
        b16 = [from_bits(to_bits(v, dtype), dtype) for v in b_vals]
        pa, pb = to_dev(pack(a16, dtype)), to_dev(pack(b16, dtype))
        po_ref = to_dev(b"\x00" * (nbytes(dtype) * n))
        po_shim = to_dev(b"\x00" * (nbytes(dtype) * n))

        # 直调（现有 Python 路径）
        for op, fn, expect in (
            ("add", lambda p, q, o: elt.add(p, q, o, n, dtype, 2.0),
             [round_at(x + 2.0 * y, dtype) for x, y in zip(a16, b16)]),
            ("sub", lambda p, q, o: elt.sub(p, q, o, n, dtype, 0.5),
             [round_at(x - 0.5 * y, dtype) for x, y in zip(a16, b16)]),
            ("mul", lambda p, q, o: elt.mul(p, q, o, n, dtype),
             [round_at(x * y, dtype) for x, y in zip(a16, b16)]),
            ("div", lambda p, q, o: elt.div(p, q, o, n, dtype),
             [round_at(x / y, dtype) for x, y in zip(a16, b16)]),
        ):
            # ---- 直调 ----
            fn(pa, pb, po_ref)
            rt.synchronize()
            direct = read(po_ref, n, dtype)

            # ---- 只按表走 shim ----
            alpha = {"add": 2.0, "sub": 0.5}.get(op, 1.0)
            shim_launch(op, dtype, "tensor", n=n, alpha=alpha,
                        ptrs={"a": pa, "b": pb, "out": po_shim})
            rt.synchronize()
            shimmed = read(po_shim, n, dtype)

            exp = pack(expect, dtype)
            total += 1
            good = shimmed == exp and direct == exp
            ok += 1 if good else 0
            tag = "" if good else "  ← 表或实现有问题"
            print(f"  {op:4s}(tensor) {dtype:4s} n={n} 直调==参照:{direct == exp} "
                  f"shim==参照:{shimmed == exp} 直调==shim:{direct == shimmed}{tag}")

        # ---- scalar 变体 ----
        po_ref2 = to_dev(b"\x00" * (nbytes(dtype) * n))
        po_shim2 = to_dev(b"\x00" * (nbytes(dtype) * n))
        for op, fn, expect in (
            ("add", lambda p, o: elt.add_scalar(p, o, n, dtype, 0.5),
             [round_at(x + 0.5, dtype) for x in a16]),
            ("sub", lambda p, o: elt.sub_scalar(p, o, n, dtype, 0.25),
             [round_at(x - 0.25, dtype) for x in a16]),
            ("mul", lambda p, o: elt.mul_scalar(p, o, n, dtype, 0.25),
             [round_at(x * 0.25, dtype) for x in a16]),
            ("div", lambda p, o: elt.div_scalar(p, o, n, dtype, 2.0),
             [round_at(x / 2.0, dtype) for x in a16]),
        ):
            fn(pa, po_ref2)
            rt.synchronize()
            direct = read(po_ref2, n, dtype)
            sv = {"add": 0.5, "sub": 0.25, "mul": 0.25, "div": 2.0}[op]
            shim_launch(op, dtype, "scalar", n=n, scalar=sv,
                        ptrs={"a": pa, "out": po_shim2})
            rt.synchronize()
            shimmed = read(po_shim2, n, dtype)
            exp = pack(expect, dtype)
            total += 1
            good = shimmed == exp and direct == exp
            ok += 1 if good else 0
            print(f"  {op:4s}(scalar) {dtype:4s} n={n} 直调==参照:{direct == exp} "
                  f"shim==参照:{shimmed == exp}{'' if good else '  ← 有问题'}")
        for p in (pa, pb, po_ref, po_shim, po_ref2, po_shim2):
            rt.free(p)

    print("=== §3 alpha 语义：mul/div 传 alpha=0 也必须与不传一致（表里 alpha_effective=False）===")
    n = 1024
    pa = to_dev(pack([0.5] * n, "fp32"))
    pb = to_dev(pack([4.0] * n, "fp32"))
    po1 = to_dev(b"\x00" * (4 * n))
    po2 = to_dev(b"\x00" * (4 * n))
    for op in ("mul", "div"):
        shim_launch(op, "fp32", "tensor", n=n, alpha=1.0,
                    ptrs={"a": pa, "b": pb, "out": po1})
        shim_launch(op, "fp32", "tensor", n=n, alpha=0.0,
                    ptrs={"a": pa, "b": pb, "out": po2})
        rt.synchronize()
        same = read(po1, n, "fp32") == read(po2, n, "fp32")
        total += 1
        ok += 1 if same else 0
        print(f"  {op}: alpha=1.0 与 alpha=0.0 结果相同={same}"
              f"（表里 alpha_effective={elt.fast_op(op, 'fp32')['alpha_effective']}）")

    print("=== §4 get_launch_meta：C++ 侧 load_kernel 需要的东西一次拿全 ===")
    for op, variant in (("add", "tensor"), ("add", "scalar"), ("mul", "tensor")):
        m = elt.get_launch_meta(op, "fp16", variant)
        print(f"  {op}({variant}) cubin={len(m['cubin']):5d} B name={m['name']!r} "
              f"warps={m['num_warps']} stages={m['num_stages']} shared={m['shared']} "
              f"args={m['args']} n_ptrs={m['n_ptrs']} n_i32={m['n_i32']} n_f32={m['n_f32']}")
        total += 1
        good = len(m["cubin"]) > 0 and len(m["args"]) == (m["n_ptrs"] + m["n_i32"] + m["n_f32"])
        ok += 1 if good else 0
        if not good:
            print("      ← cubin 为空或 args 计数不一致")

    print("[end] torch_modules =", len(torchmods()))
    verdict = "PASS" if ok == total else "FAIL"
    print(f"{ok}/{total}")
    print("FASTOP_VERDICT", verdict)
    return 0 if verdict == "PASS" else 1


def round_at(v, dtype):
    return from_bits(to_bits(v, dtype), dtype)


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 2. 入口 + shim 模拟器验收 ==="
timeout --foreground -k 5 300 "$PY" -B -u "$W/fastop_api.py"
PYRC=$?
echo "PY_RC=$PYRC"

echo "=== 3. 卡还在吗 ==="
if timeout --foreground -k 2 20 "$PY_VENDOR" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE199_DONE py_rc=$PYRC"
exit "$PYRC"
