set -u
# 目的（第 3 步 (a)(b)）：证明「corex-torch 可以完全不借 torch，自己用 ctypes 直连
# 厂商 libcuda 建 context、拿设备属性、分配显存并做逐字节一致的 H2D/D2H 往返」。
#
# 为什么这件事必须由我们自己证明：corex-triton 的运行期把 context 与 stream 全挂在
# 厂商 torch 2.1 的 torch.cuda 上（src/triton_corex/driver.py 的 _ensure_context /
# CorexUtils.get_current_stream / CoreLauncher 的 global scratch），而 corex-torch 的
# 上游是 CPU 版 torch 2.8，没有可用的 torch.cuda。这条探针是那条接缝的第一步：
# 只 import ctypes，不 import torch，也不 import triton。
#
# 为什么用 ~/venvs/corex28 的 python：那是 corex-torch 的目标运行时（隔离 venv，
# 里面连 torch 都还没装），用它跑等于顺手证明「目标运行时本身够用」。
#
# 为什么先验卡：corex-triton/docs/00-FACTS.md §5 记录过裸 cuCtxCreate 在坏卡上挂死
# 15 分钟。沙箱 runner 已经把健康卡写进 CUDA_VISIBLE_DEVICES（子进程里重编号成
# 0..N-1），本脚本**不覆盖**它——覆盖成 0 会跑到物理 0 号卡上。
#
# 跑法：python tools/remote.py sbx-run probes/150_noctx_runtime.sh --name 150 --tail 0
#
# 判定：末行 PROBE150_VERDICT PASS 才算过。

PY_ISO=${HOME}/venvs/corex28/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=${HOME}/coffin-work/base/sandbox
W=$S/work
T=$W/tb

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
[ -x "$PY_ISO" ] || { echo "MISSING_PY $PY_ISO"; exit 66; }
echo "PY_ISO=$PY_ISO"

cat > "$T/150_ctx.py" <<'PYEOF'
"""第 3 步 (a)(b)：只用 ctypes 直连 libcuda，建 context、读属性、验内存往返。

不 import torch / numpy / triton。所有 cu* 调用的返回值都打印出来，
因为厂商库可能只实现了一部分符号，猜出来的常量与编号都必须靠 rc 证实。
"""

import ctypes
import os
import sys
import time

COREX = os.environ.get("COREX_HOME", "/usr/local/corex")
LIBPATH = os.path.join(COREX, "lib64", "libcuda.so.1")

#: CUDA 10.2 的 CUdevice_attribute 编号（cuda.h）。厂商是否逐个实现 UNKNOWN，
#: 所以每个属性都连 rc 一起打印：rc != 0 一律不当作 0 看。
ATTRS = [
    (1, "MAX_THREADS_PER_BLOCK"),
    (2, "MAX_BLOCK_DIM_X"),
    (3, "MAX_BLOCK_DIM_Y"),
    (4, "MAX_BLOCK_DIM_Z"),
    (5, "MAX_GRID_DIM_X"),
    (8, "MAX_SHARED_MEMORY_PER_BLOCK"),
    (10, "WARP_SIZE"),
    (12, "MAX_REGISTERS_PER_BLOCK"),
    (13, "CLOCK_RATE_kHz"),
    (16, "MULTIPROCESSOR_COUNT"),
    (31, "CONCURRENT_KERNELS"),
    (36, "MEMORY_CLOCK_RATE_kHz"),
    (37, "GLOBAL_MEMORY_BUS_WIDTH"),
    (38, "L2_CACHE_SIZE_B"),
    (39, "MAX_THREADS_PER_MULTIPROCESSOR"),
    (41, "UNIFIED_ADDRESSING"),
    (75, "COMPUTE_CAPABILITY_MAJOR"),
    (76, "COMPUTE_CAPABILITY_MINOR"),
    (81, "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"),
    (82, "MAX_REGISTERS_PER_MULTIPROCESSOR"),
    (97, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"),
]


def errstr(lib, rc):
    """把 CUresult 翻成人话；失败就只回数字，绝不猜。"""
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
    """按名字取符号并钉死签名。缺符号返回 None（厂商没实现，是可报告的事实）。"""
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def bind_v2(lib, base, argtypes, restype=ctypes.c_int):
    """CUDA 10 之后多数符号只剩 ``_v2`` 后缀版；两个都试并报告用的是哪个。"""
    fn = bind(lib, base + "_v2", argtypes, restype)
    if fn is not None:
        return fn, base + "_v2"
    fn = bind(lib, base, argtypes, restype)
    if fn is not None:
        return fn, base
    return None, None


def main():
    print("=== 环境 ===")
    print("python", sys.version.split()[0], sys.executable)
    print("libcuda", LIBPATH, "exists=", os.path.exists(LIBPATH))
    print("torch_in_sys_modules(before)", "torch" in sys.modules)

    lib = ctypes.CDLL(LIBPATH)

    f_init = bind(lib, "cuInit", [ctypes.c_uint])
    f_count = bind(lib, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
    f_dev = bind(lib, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    f_name = bind(lib, "cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int])
    f_attr = bind(lib, "cuDeviceGetAttribute",
                  [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int])
    f_drvver = bind(lib, "cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)])
    f_totmem, sym_totmem = bind_v2(lib, "cuDeviceTotalMem",
                                   [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int])
    f_pretain, sym_pretain = bind_v2(lib, "cuDevicePrimaryCtxRetain",
                                     [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    f_setcur, sym_setcur = bind_v2(lib, "cuCtxSetCurrent", [ctypes.c_void_p])
    f_getcur = bind(lib, "cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)])
    f_apiver = bind(lib, "cuCtxGetApiVersion",
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)])
    f_sync = bind(lib, "cuCtxSynchronize", [])
    f_alloc, sym_alloc = bind_v2(lib, "cuMemAlloc",
                                 [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    f_free, sym_free = bind_v2(lib, "cuMemFree", [ctypes.c_uint64])
    f_h2d, sym_h2d = bind_v2(lib, "cuMemcpyHtoD",
                             [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
    f_d2h, sym_d2h = bind_v2(lib, "cuMemcpyDtoH",
                             [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
    f_meminfo, sym_meminfo = bind_v2(lib, "cuMemGetInfo",
                                     [ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t)])
    f_pdestroy, sym_pdestroy = bind_v2(lib, "cuDevicePrimaryCtxRelease", [ctypes.c_int])

    missing = [n for n, f in (
        ("cuInit", f_init), ("cuDeviceGetCount", f_count), ("cuDeviceGet", f_dev),
        ("cuDeviceGetName", f_name), ("cuDeviceGetAttribute", f_attr),
        ("cuDevicePrimaryCtxRetain", f_pretain), ("cuCtxSetCurrent", f_setcur),
        ("cuCtxGetCurrent", f_getcur), ("cuCtxSynchronize", f_sync),
        ("cuMemAlloc", f_alloc), ("cuMemcpyHtoD", f_h2d), ("cuMemcpyDtoH", f_d2h),
    ) if f is None]
    if missing:
        print("MISSING_SYMBOLS", missing)
        return 4
    print("symbols=ok variants", {
        "cuDeviceTotalMem": sym_totmem, "cuDevicePrimaryCtxRetain": sym_pretain,
        "cuMemAlloc": sym_alloc, "cuMemcpyHtoD": sym_h2d, "cuMemcpyDtoH": sym_d2h,
        "cuMemGetInfo": sym_meminfo, "cuDevicePrimaryCtxRelease": sym_pdestroy,
    })

    # ---- cuInit ----
    rc = f_init(0)
    print("cuInit(0) rc=", errstr(lib, rc))
    if rc != 0:
        return 4

    n = ctypes.c_int(0)
    rc = f_count(ctypes.byref(n))
    print("cuDeviceGetCount rc=%s count=%d" % (errstr(lib, rc), n.value))
    if n.value < 1:
        print("NO_VISIBLE_DEVICE（CUDA_VISIBLE_DEVICES=%r）" % os.environ.get("CUDA_VISIBLE_DEVICES"))
        return 5

    if f_drvver is not None:
        v = ctypes.c_int(0)
        rc = f_drvver(ctypes.byref(v))
        print("cuDriverGetVersion rc=%s version=%d（10020 即 10.2）" % (errstr(lib, rc), v.value))

    dev = ctypes.c_int(0)
    rc = f_dev(ctypes.byref(dev), 0)
    print("cuDeviceGet(0) rc=%s dev=%d" % (errstr(lib, rc), dev.value))

    buf = ctypes.create_string_buffer(256)
    rc = f_name(buf, 256, dev)
    print("cuDeviceGetName rc=%s name=%r" % (errstr(lib, rc), buf.value.decode(errors="replace")))

    if f_totmem is not None:
        sz = ctypes.c_size_t(0)
        rc = f_totmem(ctypes.byref(sz), dev)
        print("cuDeviceTotalMem rc=%s bytes=%d (%.1f GiB)"
              % (errstr(lib, rc), sz.value, sz.value / float(1 << 30)))

    # ---- (a) 设备属性 ----
    print("=== (a) cuDeviceGetAttribute ===")
    good = {}
    for aid, nm in ATTRS:
        v = ctypes.c_int(-1)
        rc = f_attr(ctypes.byref(v), aid, dev)
        if rc == 0:
            good[nm] = v.value
            print("  %-38s id=%-3d = %d" % (nm, aid, v.value))
        else:
            print("  %-38s id=%-3d rc=%s" % (nm, aid, errstr(lib, rc)))

    # ---- 建 context：primary context（与 CUDA runtime / torch 走的是同一条）----
    print("=== context ===")
    ctx = ctypes.c_void_p()
    rc = f_pretain(ctypes.byref(ctx), dev)
    print("cuDevicePrimaryCtxRetain rc=%s ctx=%#x" % (errstr(lib, rc), ctx.value or 0))
    if rc != 0 or not ctx.value:
        return 6
    rc = f_setcur(ctx)
    print("cuCtxSetCurrent rc=%s" % errstr(lib, rc))
    cur = ctypes.c_void_p()
    rc = f_getcur(ctypes.byref(cur))
    print("cuCtxGetCurrent rc=%s cur=%#x same_as_retained=%s"
          % (errstr(lib, rc), cur.value or 0, bool(cur.value == ctx.value)))
    if f_apiver is not None:
        api = ctypes.c_uint(0)
        rc = f_apiver(ctx, ctypes.byref(api))
        print("cuCtxGetApiVersion rc=%s api=%d（10020 即 10.2）" % (errstr(lib, rc), api.value))

    if f_meminfo is not None:
        free_b = ctypes.c_size_t(0)
        tot_b = ctypes.c_size_t(0)
        rc = f_meminfo(ctypes.byref(free_b), ctypes.byref(tot_b))
        print("cuMemGetInfo rc=%s free=%.1f MiB total=%.1f MiB"
              % (errstr(lib, rc), free_b.value / 1048576.0, tot_b.value / 1048576.0))

    # ---- (b) 显存分配 + 逐字节往返 ----
    print("=== (b) cuMemAlloc / H2D / D2H ===")
    ok_all = True
    for nbytes in (4096, 1 << 20, 1237):
        pattern = bytes((i * 7 + 13) & 0xFF for i in range(nbytes))
        dptr = ctypes.c_uint64(0)
        rc = f_alloc(ctypes.byref(dptr), nbytes)
        if rc != 0:
            print("  n=%-8d cuMemAlloc FAILED %s" % (nbytes, errstr(lib, rc)))
            ok_all = False
            continue
        rc_h = f_h2d(dptr.value, pattern, nbytes)
        rc_s = f_sync()
        out = ctypes.create_string_buffer(nbytes)
        t0 = time.perf_counter()
        rc_d = f_d2h(out, dptr.value, nbytes)
        t1 = time.perf_counter()
        rc_s2 = f_sync()
        back = out.raw
        same = (back == pattern)
        diff_at = next((i for i in range(nbytes) if i >= len(back) or back[i] != pattern[i]), None)
        print("  n=%-8d ptr=%#x align256=%s h2d=%s sync=%s d2h=%s sync2=%s equal=%s first_diff=%s d2h=%.3fms"
              % (nbytes, dptr.value, "yes" if dptr.value % 256 == 0 else "no",
                 errstr(lib, rc_h), errstr(lib, rc_s), errstr(lib, rc_d), errstr(lib, rc_s2),
                 same, "none" if diff_at is None else diff_at, (t1 - t0) * 1e3))
        ok_all = ok_all and same and rc_h == 0 and rc_d == 0
        if f_free is not None:
            f_free(dptr.value)

    # ---- 同一地址再分配一次：确认 free 有效（否则第二次会在首次分配旁边继续吃显存）----
    d2 = ctypes.c_uint64(0)
    rc = f_alloc(ctypes.byref(d2), 1 << 20)
    print("re-alloc 1MiB rc=%s ptr=%#x" % (errstr(lib, rc), d2.value))
    if rc == 0 and f_free is not None:
        f_free(d2.value)

    print("torch_in_sys_modules(after)", "torch" in sys.modules)
    print("attrs_ok=%d attrs_failed=%d" % (len(good), len(ATTRS) - len(good)))

    verdict = (
        "PASS" if (ok_all and "WARP_SIZE" in good and good.get("COMPUTE_CAPABILITY_MAJOR") is not None)
        else "FAIL"
    )
    print("PROBE150_VERDICT", verdict)
    return 0 if verdict == "PASS" else 7


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 跑 (a)(b)：纯 ctypes，无 torch ==="
timeout --foreground -k 5 120 "$PY_ISO" -B -u "$T/150_ctx.py"
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

if [ "$RC" -eq 0 ]; then echo "PROBE150_DONE rc=0"; else echo "PROBE150_DONE rc=$RC"; fi
exit "$RC"
