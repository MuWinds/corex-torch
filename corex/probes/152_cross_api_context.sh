set -u
# 目的（跨 API 的同一个 context）：corex-torch 的设备层走 **libcudart**（cudaSetDevice /
# cudaMalloc / cudaStreamCreate），corex-triton 的 launcher 走 **libcuda 驱动 API**
# （cuModuleLoadData / cuLaunchKernel）。这两套是不是同一个 primary context，决定
# 设备层与算子层能不能在一个进程里直接对接、要不要互传 context 句柄。
#
# 要回答的四件事：
#   1) cudart 建出来的 context，driver API 的 cuCtxGetCurrent 能不能看见？句柄是否等于
#      cuDevicePrimaryCtxRetain 返回的那个？
#   2) 在这个 context 里用 driver API 装载 cubin 并 cuLaunchKernel，stream 分别用
#      NULL 与 cudaStreamCreate 出来的流（直接当 CUstream 传）；
#   3) cudaMalloc 的指针能不能被 cuMemcpyDtoH 读、cuMemAlloc 的指针能不能被 cudaMemcpy 读；
#   4) 两种创建顺序（runtime 先 / driver 先）结论是否一样 —— 即「谁先建谁赢」还是无所谓。
#
# 为什么这条最要紧：如果 1) 的句柄相同且 2) 能启动，则 corex-torch **不需要自己建 context**，
# 也不需要往算子层传 context 句柄，两层天然共享同一个 primary context。
#
# 载荷：$W/tb/tb_src.tgz（含 triton_corex/），由
#   tar czf tb_src.tgz -C corex-triton/src triton_corex
# 上传；编译链要 import triton，所以用 triton34 的 venv。
#
# 跑法：python tools/remote.py sbx-run probes/152_cross_api_context.sh --name 152 --tail 0
# 判定：末行 PROBE152_VERDICT PASS（两种顺序都 PASS 才算彻底）。

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

cat > "$T/152_cross.py" <<'PYEOF'
"""libcudart 与 libcuda 是否共享同一个 primary context —— 一条探针两个顺序。

MODE=rt2drv：先 cudart（cudaSetDevice + cudaMalloc）建 context，再用 driver API 查/装/启。
MODE=drv2rt：先 driver API（cuDevicePrimaryCtxRetain + cuCtxSetCurrent），再用 cudart。
两边都跑一遍才能回答「谁先建谁赢」。
"""

import ctypes
import hashlib
import os
import struct
import sys

T = os.environ["TB_W"]
sys.path.insert(0, os.path.join(T, "src"))
MODE = os.environ.get("MODE", "rt2drv")

CX = os.environ.get("COREX_HOME", "/usr/local/corex")
RT_PATH = os.path.join(CX, "lib64", "libcudart.so.10.2")
DRV_PATH = os.path.join(CX, "lib64", "libcuda.so.1")


def torchmods():
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def err_drv(lib, rc):
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


def err_rt(lib, rc):
    if rc == 0:
        return "0(no error)"
    try:
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        s = lib.cudaGetErrorString(ctypes.c_int(rc))
        if s:
            return "%d(%s)" % (rc, s.decode(errors="replace"))
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


# --------------------------------------------------------------------------- #
# 编译（无 torch 进程）
# --------------------------------------------------------------------------- #
def compile_vecadd(n, block):
    import triton
    import triton.language as tl
    import triton_corex  # noqa: F401  导入即注册 corex 后端
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
    kernel = triton.compile(src, target=GPUTarget("corex", 70, 32))
    print("compile torch_modules=%d cubin_bytes=%d name=%r num_warps=%d shared=%d"
          % (len(torchmods()), len(kernel.asm["cubin"]), kernel.metadata.name,
             kernel.metadata.num_warps, kernel.metadata.shared))
    return kernel


def main():
    print("MODE=%s torch_modules=%d" % (MODE, len(torchmods())))
    drv = ctypes.CDLL(DRV_PATH)
    rt = ctypes.CDLL(RT_PATH)
    print("libs libcuda=%s libcudart=%s" % (DRV_PATH, RT_PATH))

    # driver API
    d_init = bind(drv, "cuInit", [ctypes.c_uint])
    d_getcur = bind(drv, "cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)])
    d_getdev = bind(drv, "cuCtxGetDevice", [ctypes.POINTER(ctypes.c_int)])
    d_devget = bind(drv, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    d_pretain, sym_pre = bind_v2(drv, "cuDevicePrimaryCtxRetain",
                                 [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    d_setcur, sym_set = bind_v2(drv, "cuCtxSetCurrent", [ctypes.c_void_p])
    d_load = bind(drv, "cuModuleLoadData", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p])
    d_getfn = bind(drv, "cuModuleGetFunction",
                   [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p])
    d_launch = bind(drv, "cuLaunchKernel", [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    d_sync = bind(drv, "cuCtxSynchronize", [])
    d_alloc, sym_alloc = bind_v2(drv, "cuMemAlloc",
                                 [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    d_h2d, _ = bind_v2(drv, "cuMemcpyHtoD",
                       [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
    d_d2h, _ = bind_v2(drv, "cuMemcpyDtoH",
                       [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
    d_free, _ = bind_v2(drv, "cuMemFree", [ctypes.c_uint64])

    # runtime API
    r_devcount = bind(rt, "cudaGetDeviceCount", [ctypes.POINTER(ctypes.c_int)])
    r_setdev = bind(rt, "cudaSetDevice", [ctypes.c_int])
    r_getdev = bind(rt, "cudaGetDevice", [ctypes.POINTER(ctypes.c_int)])
    r_malloc = bind(rt, "cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
    r_free = bind(rt, "cudaFree", [ctypes.c_void_p])
    r_memcpy = bind(rt, "cudaMemcpy",
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int])
    r_strmcreate = bind(rt, "cudaStreamCreate", [ctypes.POINTER(ctypes.c_void_p)])
    r_strmdestroy = bind(rt, "cudaStreamDestroy", [ctypes.c_void_p])
    r_sync = bind(rt, "cudaDeviceSynchronize", [])
    r_lasterr = bind(rt, "cudaGetLastError", [])

    need = {"drv": [("cuInit", d_init), ("cuCtxGetCurrent", d_getcur),
                    ("cuCtxGetDevice", d_getdev), ("cuModuleLoadData", d_load),
                    ("cuModuleGetFunction", d_getfn), ("cuLaunchKernel", d_launch)],
            "rt": [("cudaGetDeviceCount", r_devcount), ("cudaSetDevice", r_setdev),
                   ("cudaMalloc", r_malloc), ("cudaMemcpy", r_memcpy),
                   ("cudaStreamCreate", r_strmcreate)]}
    missing = [n for grp in need.values() for n, f in grp if f is None]
    if missing:
        print("MISSING_SYMBOLS", missing)
        return 4
    print("driver_variants", {"primary_ctx": sym_pre, "set_cur": sym_set, "alloc": sym_alloc})

    n = int(os.environ.get("N", "4096"))
    block = 128
    host_a = [1.5 + 0.25 * i for i in range(n)]
    host_b = [-0.5 + 0.125 * i for i in range(n)]
    buf_a = struct.pack("<%df" % n, *host_a)
    buf_b = struct.pack("<%df" % n, *host_b)
    buf_ref = struct.pack("<%df" % n, *[float(x + y) for x, y in zip(host_a, host_b)])

    kernel = compile_vecadd(n, block)
    cubin = kernel.asm["cubin"]
    kname = kernel.metadata.name

    def current_ctx(tag):
        c = ctypes.c_void_p()
        rc = d_getcur(ctypes.byref(c))
        dev = ctypes.c_int(-1)
        rc2 = d_getdev(ctypes.byref(dev))
        print("[%s] cuCtxGetCurrent rc=%s ctx=%s  cuCtxGetDevice rc=%s dev=%d"
              % (tag, err_drv(drv, rc), ("%#x" % c.value) if c.value else "NULL",
                 err_drv(drv, rc2), dev.value))
        return c.value

    print("=== 0. 任何调用之前 ===")
    rc = d_init(0)
    print("cuInit(0) rc=%s" % err_drv(drv, rc))
    ctx0 = current_ctx("before")

    retained = None
    if MODE == "drv2rt":
        print("=== 1. 先 driver API 建 context ===")
        dev = ctypes.c_int(0)
        print("cuDeviceGet(0) rc=%s" % err_drv(drv, d_devget(ctypes.byref(dev), 0)))
        c = ctypes.c_void_p()
        print("cuDevicePrimaryCtxRetain rc=%s ctx=%#x"
              % (err_drv(drv, d_pretain(ctypes.byref(c), dev)), c.value or 0))
        retained = c.value
        print("cuCtxSetCurrent rc=%s" % err_drv(drv, d_setcur(c)))
        current_ctx("after_retain")
        print("=== 1b. 再用 cudart ===")
        cnt = ctypes.c_int(0)
        print("cudaGetDeviceCount rc=%s count=%d"
              % (err_rt(rt, r_devcount(ctypes.byref(cnt))), cnt.value))
        print("cudaSetDevice(0) rc=%s" % err_rt(rt, r_setdev(0)))
        gd = ctypes.c_int(-1)
        print("cudaGetDevice rc=%s dev=%d" % (err_rt(rt, r_getdev(ctypes.byref(gd))), gd.value))
    else:
        print("=== 1. 先 cudart 建 context ===")
        cnt = ctypes.c_int(0)
        print("cudaGetDeviceCount rc=%s count=%d"
              % (err_rt(rt, r_devcount(ctypes.byref(cnt))), cnt.value))
        print("cudaSetDevice(0) rc=%s" % err_rt(rt, r_setdev(0)))
        current_ctx("after_setdevice")          # 先看 SetDevice 自己建不建 context
        gd = ctypes.c_int(-1)
        print("cudaGetDevice rc=%s dev=%d" % (err_rt(rt, r_getdev(ctypes.byref(gd))), gd.value))

    # runtime 分配三个缓冲区：设备层负责内存
    nbytes = n * 4
    ptrs = []
    for label, payload in (("a", buf_a), ("b", buf_b), ("c", b"\x00" * nbytes)):
        p = ctypes.c_void_p()
        rc = r_malloc(ctypes.byref(p), nbytes)
        if rc != 0:
            print("cudaMalloc(%s) FAILED %s" % (label, err_rt(rt, rc)))
            return 5
        if payload is not None:
            rc2 = r_memcpy(p, payload, nbytes, 1)                    # 1 = H2D
            if rc2 != 0:
                print("cudaMemcpy H2D(%s) FAILED %s" % (label, err_rt(rt, rc2)))
                return 5
        ptrs.append(p.value)
        print("cudaMalloc(%s) rc=%s ptr=%#x" % (label, err_rt(rt, rc), p.value))
    print("=== 1c. cudart 分配之后，driver 侧看见的 context ===")
    ctx_rt = current_ctx("after_cudaMalloc")

    # 主判定：cuDevicePrimaryCtxRetain 的句柄 vs runtime 建出来的那个
    dev = ctypes.c_int(0)
    d_devget(ctypes.byref(dev), 0)
    c = ctypes.c_void_p()
    rc = d_pretain(ctypes.byref(c), dev)
    retained = c.value
    print("cuDevicePrimaryCtxRetain rc=%s ctx=%#x" % (err_drv(drv, rc), retained or 0))
    same = (ctx_rt != 0) and (retained == ctx_rt)
    print("SAME_PRIMARY_CONTEXT=%s  runtime_ctx=%s retained=%s"
          % (same, ("%#x" % ctx_rt) if ctx_rt else "NULL", "%#x" % (retained or 0)))
    print("HAD_CONTEXT_BEFORE_RUNTIME=%s" % bool(ctx0))

    # ---- 2. driver API 装载 + 启动（在 runtime 建出来的 context 里）----
    print("=== 2. driver API 装载 + 启动 ===")
    mod = ctypes.c_void_p()
    rc = d_load(ctypes.byref(mod), ctypes.c_char_p(cubin))
    print("cuModuleLoadData rc=%s mod=%#x" % (err_drv(drv, rc), mod.value or 0))
    if rc != 0:
        print("PROBE152_VERDICT FAIL")
        return 6
    fn = ctypes.c_void_p()
    rc = d_getfn(ctypes.byref(fn), mod, kname.encode())
    print("cuModuleGetFunction(%r) rc=%s fn=%#x" % (kname, err_drv(drv, rc), fn.value or 0))
    if rc != 0:
        print("PROBE152_VERDICT FAIL")
        return 6

    objs = [ctypes.c_uint64(ptrs[0]), ctypes.c_uint64(ptrs[1]), ctypes.c_uint64(ptrs[2]),
            ctypes.c_int32(n), ctypes.c_uint64(0)]
    params = (ctypes.c_void_p * len(objs))(
        *[ctypes.cast(ctypes.byref(o), ctypes.c_void_p) for o in objs])
    grid = ((n + block - 1) // block, 1, 1)
    blk = (kernel.metadata.num_warps * kernel.metadata.target.warp_size, 1, 1)
    shared = kernel.metadata.shared

    stream = ctypes.c_void_p()
    rc = r_strmcreate(ctypes.byref(stream))
    print("cudaStreamCreate rc=%s stream=%#x" % (err_rt(rt, rc), stream.value or 0))

    launches = [("null", None)]
    if stream.value:
        launches.append(("cudaStream", stream))
    for label, st in launches:
        if label == "cudaStream":
            r_memcpy(ptrs[2], b"\x00" * nbytes, nbytes, 1)  # 清 c 再算一遍
        rc = d_launch(fn, grid[0], 1, 1, blk[0], 1, 1, shared, st, params, None)
        rc_s = d_sync()
        rc_rt = r_sync()
        print("cuLaunchKernel(stream=%s) rc=%s cuCtxSync rc=%s cudaDeviceSync rc=%s"
              % (label, err_drv(drv, rc), err_drv(drv, rc_s), err_rt(rt, rc_rt)))

    # ---- 3. 交叉读写：谁分配的指针谁能读 ----
    print("=== 3. 交叉读写 ===")
    out = ctypes.create_string_buffer(nbytes)
    rc = d_d2h(out, ptrs[2], nbytes)                     # driver 读 cudaMalloc 的指针
    print("cuMemcpyDtoH(cudaMalloc ptr) rc=%s" % err_drv(drv, rc))
    equal_drv = (out.raw == buf_ref)
    print("RESULT_VIA_DRIVER equal_bytes=%s" % equal_drv)

    out2 = ctypes.create_string_buffer(nbytes)
    rc = r_memcpy(out2, ctypes.c_void_p(ptrs[2]), nbytes, 2)   # runtime 读同一个指针
    print("cudaMemcpy D2H(同指针) rc=%s" % err_rt(rt, rc))
    equal_rt = (out2.raw == buf_ref)
    print("RESULT_VIA_RUNTIME equal_bytes=%s" % equal_rt)

    # 反方向：driver API 分配的指针，用 cudaMemcpy 读
    dp = ctypes.c_uint64(0)
    rc = d_alloc(ctypes.byref(dp), nbytes)
    if rc == 0:
        d_h2d(dp.value, buf_a, nbytes)
        back = ctypes.create_string_buffer(nbytes)
        rc2 = r_memcpy(back, ctypes.c_void_p(dp.value), nbytes, 2)
        print("cuMemAlloc ptr=%#x，cudaMemcpy D2H rc=%s equal=%s"
              % (dp.value, err_rt(rt, rc2), back.raw == buf_a))
        d_free(dp.value)
    else:
        print("cuMemAlloc FAILED %s" % err_drv(drv, rc))

    if r_strmdestroy is not None and stream.value:
        print("cudaStreamDestroy rc=%s" % err_rt(rt, r_strmdestroy(stream)))

    print("last_rt_error=%s" % err_rt(rt, r_lasterr()))
    print("torch_modules_end=%d" % len(torchmods()))
    ok = same and equal_drv and equal_rt
    print("PROBE152_VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 7


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== A. runtime 先建 context ==="
MODE=rt2drv timeout --foreground -k 5 150 "$PY_TRITON" -B -u "$T/152_cross.py"
RC_A=$?
echo "RC_A=$RC_A"

echo "=== B. driver 先建 context ==="
MODE=drv2rt timeout --foreground -k 5 150 "$PY_TRITON" -B -u "$T/152_cross.py"
RC_B=$?
echo "RC_B=$RC_B"

echo "=== 事后健康 ==="
if timeout --foreground -k 2 10 "$PY_VENDOR" -B -c \
    "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
    >/dev/null 2>&1; then
  echo "CARD_STILL_HEALTHY"
else
  echo "CARD_BROKEN_AFTER_PROBE"
fi

echo "PROBE152_DONE rc_a=$RC_A rc_b=$RC_B"
[ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]
