set -u
# 目的：`CorexLauncher.__call__` 把 `metadata.shared` 原样当 `cuLaunchKernel` 的
# sharedMemBytes 传下去。这张卡的 `MAX_SHARED_MEMORY_PER_BLOCK` 实测 131072（128 KiB），
# 但 CUDA 的老规矩是「**默认动态 shared 上限 48 KiB**，超过要先 cuFuncSetAttribute 抬到
# 128 KiB」。这决定算子层能不能开大 tile（softmax 一行的 BLOCK、matmul 的分块），
# 也决定 adapter 里要不要在每次 load_binary 之后补一次 cuFuncSetAttribute。
#
# 这正好是 docs/00-FACTS.md §8 里那条 UNKNOWN（「128 KB 是可申请上限还是 48 KB + opt-in」），
# 用一条探针把它关掉。
#
# 做法：用一个 shared=0 的 vecadd kernel（不需要 shared），只改 cuLaunchKernel 的
# sharedMemBytes 参数——驱动只校验「申请量 ≤ 该函数的动态 shared 上限」，所以
# 一个不占 shared 的 kernel 也能当尺子用，不必让 Triton 生成大 shared 的 kernel。
#
# 载荷：$W/tb/tb_src.tgz（含 triton_corex/）。
# 跑法：python tools/remote.py sbx-run probes/153_shared_limit.sh --name 153 --tail 0
# 判定：末行 PROBE153_VERDICT 里带 BOUNDARY（默认上限与抬升后可用的最大值都要量到）。

PY_TRITON=${HOME}/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=${HOME}/coffin-work/base/sandbox
W=$S/work
T=$W/tb
export TB_W=$T

echo "HOST=$(hostname) DATE=$(date -Is)"
echo "CVD=${CUDA_VISIBLE_DEVICES:-unset}  # runner 给的，子进程内重编号成 0..N-1"

# ---- 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES，不要自己设）----
if timeout --foreground -k 2 10 "$PY_VENDOR" -B -c \
    "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
    >/dev/null 2>&1; then
  echo "CARD_OK"
else
  echo "CARD_UNHEALTHY_OR_DEAD"
  exit 9
fi

mkdir -p "$T"
[ -x "$PY_TRITON" ] || { echo "MISSING_PY $PY_TRITON"; exit 66; }
if [ -f "$T/tb_src.tgz" ]; then
  rm -rf "$T/src" && mkdir -p "$T/src" && tar xzf "$T/tb_src.tgz" -C "$T/src" || exit 67
else
  echo "PAYLOAD_MISSING $T/tb_src.tgz"; exit 66
fi

cat > "$T/153_shared.py" <<'PYEOF'
"""动态 shared 的上限：默认 48 KiB 还是 128 KiB？超过要不要 cuFuncSetAttribute？"""

import ctypes
import os
import struct
import sys

T = os.environ["TB_W"]
sys.path.insert(0, os.path.join(T, "src"))

COREX = os.environ.get("COREX_HOME", "/usr/local/corex")
LIBPATH = os.path.join(COREX, "lib64", "libcuda.so.1")
KIB = 1024

#: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8（cuda.h，CUDA 9+ 稳定）。
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def errstr(lib, rc):
    if rc == 0:
        return "0(SUCCESS)"
    p = ctypes.c_char_p()
    try:
        lib.cuGetErrorString(ctypes.c_int(rc), ctypes.byref(p))
        if p.value:
            return "%d(%s)" % (rc, p.value.decode(errors="replace"))
    except Exception:                                    # noqa: BLE001
        pass
    return str(rc)


def bind(lib, name, argtypes, restype=ctypes.c_int):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def bind_v2(lib, base, argtypes, restype=ctypes.c_int):
    fn = bind(lib, base + "_v2", argtypes, restype)
    if fn is not None:
        return fn, base + "_v2"
    fn = bind(lib, base, argtypes, restype)
    return fn, (base if fn is not None else None)


def compile_vecadd(n, block):
    import triton
    import triton.language as tl
    import triton_corex  # noqa: F401
    from triton.backends.compiler import GPUTarget

    @triton.jit
    def vecadd(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(c_ptr + off, tl.load(a_ptr + off, mask=mask) + tl.load(b_ptr + off, mask=mask),
                 mask=mask)

    src = triton.compiler.ASTSource(
        fn=vecadd,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "n": "i32",
                   "BLOCK": "constexpr"},
        constexprs={"BLOCK": block},
    )
    k = triton.compile(src, target=GPUTarget("corex", 70, 32))
    print("compile torch_modules=%d cubin_bytes=%d shared=%d num_warps=%d"
          % (len([m for m in sys.modules if m == "torch" or m.startswith("torch.")]),
             len(k.asm["cubin"]), k.metadata.shared, k.metadata.num_warps))
    return k


def main():
    lib = ctypes.CDLL(LIBPATH)
    f_init = bind(lib, "cuInit", [ctypes.c_uint])
    f_dev = bind(lib, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    f_attr = bind(lib, "cuDeviceGetAttribute",
                  [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int])
    f_pretain, _ = bind_v2(lib, "cuDevicePrimaryCtxRetain",
                           [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    f_setcur, _ = bind_v2(lib, "cuCtxSetCurrent", [ctypes.c_void_p])
    f_alloc, _ = bind_v2(lib, "cuMemAlloc", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    f_free, _ = bind_v2(lib, "cuMemFree", [ctypes.c_uint64])
    f_load = bind(lib, "cuModuleLoadData", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p])
    f_getfn = bind(lib, "cuModuleGetFunction",
                   [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p])
    f_setfunc = bind(lib, "cuFuncSetAttribute",
                     [ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
    f_launch = bind(lib, "cuLaunchKernel", [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    f_sync = bind(lib, "cuCtxSynchronize", [])

    print("cuInit rc=%s" % errstr(lib, f_init(0)))
    dev = ctypes.c_int(0)
    f_dev(ctypes.byref(dev), 0)
    for aid, nm in ((8, "MAX_SHARED_MEMORY_PER_BLOCK"), (81, "PER_MULTIPROCESSOR"),
                    (97, "PER_BLOCK_OPTIN"), (1, "MAX_THREADS_PER_BLOCK")):
        v = ctypes.c_int(-1)
        rc = f_attr(ctypes.byref(v), aid, dev)
        print("attr %-30s id=%-3d rc=%s = %d (%.0f KiB)"
              % (nm, aid, errstr(lib, rc), v.value, v.value / 1024.0))
    ctx = ctypes.c_void_p()
    print("cuDevicePrimaryCtxRetain rc=%s" % errstr(lib, f_pretain(ctypes.byref(ctx), dev)))
    print("cuCtxSetCurrent rc=%s" % errstr(lib, f_setcur(ctx)))

    n, block = 4096, 128
    kernel = compile_vecadd(n, block)
    cubin, kname = kernel.asm["cubin"], kernel.metadata.name
    nb = n * 4
    a = [1.5 + 0.25 * i for i in range(n)]
    b = [-0.5 + 0.125 * i for i in range(n)]
    bufa = struct.pack("<%df" % n, *a)
    bufb = struct.pack("<%df" % n, *b)

    ptrs = []
    for payload in (bufa, bufb, b"\x00" * nb):
        p = ctypes.c_uint64(0)
        rc = f_alloc(ctypes.byref(p), nb)
        if rc != 0:
            print("cuMemAlloc FAILED %s" % errstr(lib, rc))
            return 5
        if payload is not None:
            bind_v2(lib, "cuMemcpyHtoD", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])[0](
                p.value, payload, nb)
        ptrs.append(p.value)

    mod = ctypes.c_void_p()
    print("cuModuleLoadData rc=%s" % errstr(lib, f_load(ctypes.byref(mod), ctypes.c_char_p(cubin))))
    fn = ctypes.c_void_p()
    print("cuModuleGetFunction rc=%s" % errstr(lib, f_getfn(ctypes.byref(fn), mod, kname.encode())))
    if not fn.value:
        print("PROBE153_VERDICT FAIL getfn")
        return 6
    print("cuFuncSetAttribute symbol present=%s" % (f_setfunc is not None))

    objs = [ctypes.c_uint64(ptrs[0]), ctypes.c_uint64(ptrs[1]), ctypes.c_uint64(ptrs[2]),
            ctypes.c_int32(n), ctypes.c_uint64(0)]
    params = (ctypes.c_void_p * len(objs))(
        *[ctypes.cast(ctypes.byref(o), ctypes.c_void_p) for o in objs])
    grid = ((n + block - 1) // block, 1, 1)
    blk = kernel.metadata.num_warps * kernel.metadata.target.warp_size

    def try_launch(shared_bytes, tag):
        rc = f_launch(fn, grid[0], 1, 1, blk, 1, 1, shared_bytes, None, params, None)
        if rc == 0:
            f_sync()
        print("  %-28s shared=%-8d (%6.1f KiB) rc=%s"
              % (tag, shared_bytes, shared_bytes / 1024.0, errstr(lib, rc)))
        return rc

    print("=== 1. 默认（未 opt-in）能申请多大动态 shared ===")
    sizes = [0, 48 * KIB, 48 * KIB + 4, 64 * KIB, 96 * KIB, 128 * KIB, 128 * KIB + 4 * KIB]
    default_ok = -1
    default_fail = None
    for s in sizes:
        rc = try_launch(s, "no-optin")
        if rc == 0:
            default_ok = s
        elif default_fail is None:
            default_fail = (s, rc)

    print("=== 2. cuFuncSetAttribute(MAX_DYNAMIC_SHARED=8) 抬到 128 KiB 再试 ===")
    optin_ok = -1
    optin_fail = None
    if f_setfunc is not None:
        for target in (128 * KIB, 100 * KIB):
            rc = f_setfunc(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, target)
            print("  cuFuncSetAttribute(8, %d = %.0f KiB) rc=%s"
                  % (target, target / 1024.0, errstr(lib, rc)))
            if rc != 0:
                continue
            for s in (48 * KIB, 96 * KIB, 100 * KIB, 128 * KIB, 128 * KIB + 4 * KIB):
                rc2 = try_launch(s, "optin=%dKiB" % (target // KIB))
                if rc2 == 0:
                    optin_ok = max(optin_ok, s)
                elif optin_fail is None:
                    optin_fail = (s, rc2)
    else:
        print("  cuFuncSetAttribute 符号不存在 → 只能靠默认上限")

    print("=== 3. 结果正确性（shared=0 那一发的输出）===")
    out = ctypes.create_string_buffer(nb)
    bind_v2(lib, "cuMemcpyDtoH", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])[0](
        out, ptrs[2], nb)
    ref = struct.pack("<%df" % n, *[float(x + y) for x, y in zip(a, b)])
    print("  equal_bytes=%s" % (out.raw == ref))

    for p in ptrs:
        f_free(p)

    print("SUMMARY default_ok=%d B (%.0f KiB)  first_fail=%s"
          % (default_ok, default_ok / 1024.0, default_fail))
    print("SUMMARY optin_ok=%d B (%.0f KiB)  first_fail=%s"
          % (optin_ok, optin_ok / 1024.0, optin_fail))
    # 判定只要求「量到了默认上限」：opt-in 是否可用是附加信息。
    ok = default_fail is not None and default_ok >= 0
    print("PROBE153_VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 7


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 153: 动态 shared 上限 ==="
timeout --foreground -k 5 180 "$PY_TRITON" -B -u "$T/153_shared.py"
RC=$?
echo "PY_RC=$RC"

echo "=== 事后健康 ==="
if timeout --foreground -k 2 10 "$PY_VENDOR" -B -c \
    "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
    >/dev/null 2>&1; then
  echo "CARD_STILL_HEALTHY"
else
  echo "CARD_BROKEN_AFTER_PROBE"
fi

echo "PROBE153_DONE rc=$RC"
exit "$RC"
