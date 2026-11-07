set -u
# 目的（第 3 步 (c) + 第 2 步的判决点）：在**一个完全不 import torch 的进程**里，
#   1) 用 triton 3.4 + corex-triton 的 triton_corex 走完 triton.compile，拿到 asm["cubin"]；
#   2) 只用 ctypes 直连 libcuda 建 context、分配显存、cuModuleLoadData + cuLaunchKernel
#      把同一个 cubin 启起来，并与 CPU 参照值逐字节比较。
#
# 为什么这是本案的判决点：如果这条链路通，说明「corex-torch 自带 runtime 层 + 复用
# corex-triton 的编译链」成立，corex-triton 里那 12 处 torch 触点就都只是
# 「换一个 provider」，不需要改 corex-triton 的源码。
#
# 为什么先编译后碰卡：编译侧（第 2 步）的结论不依赖卡，先拿到它；即使建 context 失败，
# 第 2 步的答案也已经打印出来了。
#
# 为什么先验卡：00-FACTS §5 记录裸 cuCtxCreate 在坏卡上挂死 15 分钟；本脚本继承沙箱
# runner 给的 CUDA_VISIBLE_DEVICES（子进程里重编号 0..N-1），**不覆盖**它。
#
# 载荷：$W/tb/tb_src.tgz（含 triton_corex/ 的四个 .py），由
#   tar czf tb_src.tgz -C corex-triton/src triton_corex  然后 remote.py put 上传。
#
# 跑法：python tools/remote.py sbx-run probes/151_torchfree_launch.sh --name 151 --tail 0
# 判定：PROBE151_VERDICT PASS。

PY_TRITON=${HOME}/venvs/triton34/bin/python
PY_VENDOR=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
S=${HOME}/coffin-work/base/sandbox
W=$S/work
T=$W/tb
export TB_W=$T
export COREX_DUMP_IR=$T/biir.ll

echo "HOST=$(hostname) DATE=$(date -Is)"
echo "CVD=${CUDA_VISIBLE_DEVICES:-unset}  # runner 给的，子进程内重编号成 0..N-1"

# ---- 验卡（继承 CVD）----
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
echo "PY_TRITON=$PY_TRITON"
if [ -f "$T/tb_src.tgz" ]; then
  # 每次都重解包：远端那份可能是上一轮的代码，改了不重解包会白跑一轮。
  rm -rf "$T/src" && mkdir -p "$T/src" && tar xzf "$T/tb_src.tgz" -C "$T/src" || exit 67
else
  echo "PAYLOAD_MISSING $T/tb_src.tgz"; exit 66
fi
ls "$T/src/triton_corex"/*.py | head -6

cat > "$T/151_launch.py" <<'PYEOF'
"""无 torch 进程里的编译 + 启动：第 2 步与第 3 步 (c) 共用一条探针。

证据分三段打印：
  [1] 编译：triton.compile → asm["cubin"]，并在每一步后检查 sys.modules 里有没有 torch；
  [2] context/显存：ctypes 直连 libcuda 的 cuDevicePrimaryCtxRetain + cuMemAlloc；
  [3] 启动：cuModuleLoadData + cuModuleGetFunction + cuLaunchKernel，取回结果比对。
"""

import ctypes
import hashlib
import os
import struct
import sys
import time

T = os.environ["TB_W"]
sys.path.insert(0, os.path.join(T, "src"))

COREX = os.environ.get("COREX_HOME", "/usr/local/corex")
LIBPATH = os.path.join(COREX, "lib64", "libcuda.so.1")
#: 编译目标：arch=70 只是借这个数字表达「没有 cp.async、没有 bf16 mma」，
#: 与 corex-triton/docs/00-FACTS.md §4/§8 的口径一致。
TARGET = None


def torchmods():
    """返回当前进程里已导入的 torch 相关模块名（本案要求为空）。"""
    return sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))


def say(tag):
    mods = torchmods()
    print("[%s] torch_modules=%d %s" % (tag, len(mods), mods[:8]))


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
    return fn, base if fn is not None else None


# --------------------------------------------------------------------------- #
# [1] 编译
# --------------------------------------------------------------------------- #
def compile_vecadd(n, block):
    import triton
    import triton.language as tl

    say("after_import_triton")

    import triton_corex  # noqa: F401  导入即把 corex 后端挂进 triton.backends
    from triton.backends.compiler import GPUTarget

    # 注意：这里**不调用** triton_corex.set_active()。它会 CorexDriver() →
    # GPUDriver.__init__ → import torch（见 triton/backends/driver.py:41），
    # 而本案要证明的是「编译不需要 torch」，所以故意不碰它。
    say("after_import_triton_corex")

    @triton.jit
    def vecadd(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(c_ptr + off, tl.load(a_ptr + off, mask=mask) + tl.load(b_ptr + off, mask=mask),
                 mask=mask)

    target = GPUTarget("corex", 70, 32)
    src = triton.compiler.ASTSource(
        fn=vecadd,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "n": "i32",
                   "BLOCK": "constexpr"},
        constexprs={"BLOCK": block},
    )
    t0 = time.perf_counter()
    kernel = triton.compile(src, target=target)
    dt = time.perf_counter() - t0
    say("after_compile")
    print("COMPILE_OK sec=%.3f asm_keys=%s" % (dt, sorted(kernel.asm.keys())))
    cubin = kernel.asm["cubin"]
    print("cubin_bytes=%d sha256=%s" % (len(cubin), hashlib.sha256(cubin).hexdigest()))
    md = kernel.metadata
    print("metadata name=%r num_warps=%d shared=%d num_ctas=%d global_scratch_size=%s warp_size=%d"
          % (md.name, md.num_warps, md.shared, md.num_ctas,
             getattr(md, "global_scratch_size", None), md.target.warp_size))
    # 内核参数表直接从 biir 的 define 行读，不靠猜：cuLaunchKernel 的 kernelParams
    # 必须与 cubin 的符号签名严格对齐，猜错一位就是静默读到垃圾。
    biir = kernel.asm.get("biir", "")
    define = next((ln for ln in biir.splitlines() if "iluvatar_kernel" in ln and "@" in ln), "")
    print("biir_define=%s" % define.strip())
    return kernel, cubin, md, define


# --------------------------------------------------------------------------- #
# [2] context + 显存（纯 ctypes）
# --------------------------------------------------------------------------- #
def open_context():
    lib = ctypes.CDLL(LIBPATH)
    f_init = bind(lib, "cuInit", [ctypes.c_uint])
    f_dev = bind(lib, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    f_pretain, sym_pre = bind_v2(lib, "cuDevicePrimaryCtxRetain",
                                 [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    f_setcur, sym_set = bind_v2(lib, "cuCtxSetCurrent", [ctypes.c_void_p])
    f_alloc, sym_alloc = bind_v2(lib, "cuMemAlloc",
                                 [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    f_free, sym_free = bind_v2(lib, "cuMemFree", [ctypes.c_uint64])
    f_h2d, sym_h2d = bind_v2(lib, "cuMemcpyHtoD",
                             [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
    f_d2h, sym_d2h = bind_v2(lib, "cuMemcpyDtoH",
                             [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
    f_sync = bind(lib, "cuCtxSynchronize", [])
    print("ctx_symbols", {"primary_ctx": sym_pre, "set_cur": sym_set, "alloc": sym_alloc,
                          "free": sym_free, "h2d": sym_h2d, "d2h": sym_d2h})
    rc = f_init(0)
    print("cuInit(0) rc=%s" % errstr(lib, rc))
    dev = ctypes.c_int(0)
    rc = f_dev(ctypes.byref(dev), 0)
    print("cuDeviceGet(0) rc=%s dev=%d" % (errstr(lib, rc), dev.value))
    ctx = ctypes.c_void_p()
    rc = f_pretain(ctypes.byref(ctx), dev)
    print("cuDevicePrimaryCtxRetain rc=%s ctx=%#x" % (errstr(lib, rc), ctx.value or 0))
    rc = f_setcur(ctx)
    print("cuCtxSetCurrent rc=%s" % errstr(lib, rc))
    return dict(lib=lib, alloc=f_alloc, free=f_free, h2d=f_h2d, d2h=f_d2h, sync=f_sync,
                errstr=lambda rc: errstr(lib, rc))


def mem_alloc(d, nbytes):
    p = ctypes.c_uint64(0)
    rc = d["alloc"](ctypes.byref(p), nbytes)
    if rc != 0:
        raise RuntimeError("cuMemAlloc(%d) 失败 %s" % (nbytes, d["errstr"](rc)))
    return p.value


# --------------------------------------------------------------------------- #
# [3] 装载 + 启动
# --------------------------------------------------------------------------- #
def launch(d, cubin, md, kname, nparams, params_objs, grid, block, shared):
    lib = d["lib"]
    f_load = bind(lib, "cuModuleLoadData", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p])
    f_getfn = bind(lib, "cuModuleGetFunction",
                   [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p])
    f_launch = bind(lib, "cuLaunchKernel", [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ])
    f_stream, sym_stream = bind_v2(lib, "cuStreamCreate",
                                   [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint])

    mod = ctypes.c_void_p()
    rc = f_load(ctypes.byref(mod), ctypes.c_char_p(cubin))
    print("cuModuleLoadData rc=%s mod=%#x" % (errstr(lib, rc), mod.value or 0))
    if rc != 0:
        return rc, "load_failed"
    fn = ctypes.c_void_p()
    rc = f_getfn(ctypes.byref(fn), mod, kname.encode())
    print("cuModuleGetFunction(%r) rc=%s fn=%#x" % (kname, errstr(lib, rc), fn.value or 0))
    if rc != 0:
        return rc, "getfn_failed"

    params = (ctypes.c_void_p * nparams)(
        *[ctypes.cast(ctypes.byref(o), ctypes.c_void_p) for o in params_objs])

    # 先用 NULL（legacy default stream）启动：先证明「连 stream 句柄都不需要」；
    # 再用自己 cuStreamCreate 出来的句柄重来一遍，证明适配层能提供 stream。
    streams = [("null", None)]
    if f_stream is not None:
        sp = ctypes.c_void_p()
        rc = f_stream(ctypes.byref(sp), 0)
        print("cuStreamCreate(%s) rc=%s stream=%#x" % (sym_stream, errstr(lib, rc), sp.value or 0))
        if rc == 0:
            streams.append(("own", sp))

    results = []
    for label, stream in streams:
        rc = f_launch(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                      shared, stream, params, None)
        rc_s = d["sync"]()
        print("cuLaunchKernel(stream=%s) rc=%s sync_rc=%s" % (label, errstr(lib, rc), d["errstr"](rc_s)))
        results.append((label, rc))
    return 0, "launched"


def main():
    say("start")
    n = int(os.environ.get("N", "4096"))
    block = int(os.environ.get("BLOCK", "128"))

    # 用可在 fp32 里精确表示的输入，判定就能要求「逐字节相等」而不是「误差够小」。
    host_a = [1.5 + 0.25 * i for i in range(n)]
    host_b = [-0.5 + 0.125 * i for i in range(n)]
    host_ref = [float(x + y) for x, y in zip(host_a, host_b)]
    buf_a = struct.pack("<%df" % n, *host_a)
    buf_b = struct.pack("<%df" % n, *host_b)
    buf_ref = struct.pack("<%df" % n, *host_ref)

    kernel, cubin, md, define = compile_vecadd(n, block)
    kname = md.name

    # biir 的参数表就是 cubin 的符号签名。vecadd 有 4 个用户参数，
    # Triton 3.4 末尾多一个 global scratch 指针（见 corex-triton driver.py:186-192）。
    nparams = define.count(",") + 1 if define else 0
    print("kernel_params=%d expect=5 (4 user + global scratch)" % nparams)
    if nparams != 5:
        print("UNEXPECTED_PARAM_COUNT 需要在启动前重新核对参数表")
        print("PROBE151_VERDICT FAIL")
        return 8

    d = open_context()
    pa = mem_alloc(d, n * 4)
    pb = mem_alloc(d, n * 4)
    pc = mem_alloc(d, n * 4)
    print("device_ptrs a=%#x b=%#x c=%#x bytes_each=%d" % (pa, pb, pc, n * 4))
    print("h2d a rc=%s" % d["errstr"](d["h2d"](pa, buf_a, n * 4)))
    print("h2d b rc=%s" % d["errstr"](d["h2d"](pb, buf_b, n * 4)))
    print("memset c rc=%s" % d["errstr"](d["h2d"](pc, b"\x00" * (n * 4), n * 4)))
    d["sync"]()

    # 参数按值的小对象：cuLaunchKernel 要的是「指向参数值」的指针数组。
    objs = [ctypes.c_uint64(pa), ctypes.c_uint64(pb), ctypes.c_uint64(pc),
            ctypes.c_int32(n), ctypes.c_uint64(0)]
    grid = ((n + block - 1) // block, 1, 1)
    blk = (md.num_warps * md.target.warp_size, 1, 1)
    print("launch grid=%s block=%s shared=%d" % (grid, blk, md.shared))
    rc, stage = launch(d, cubin, md, kname, nparams, objs, grid, blk, md.shared)
    if rc != 0:
        print("LAUNCH_STAGE", stage)
        print("PROBE151_VERDICT FAIL")
        return 9

    out = ctypes.create_string_buffer(n * 4)
    rc = d["d2h"](out, pc, n * 4)
    print("d2h c rc=%s" % d["errstr"](rc))
    got = out.raw
    equal = got == buf_ref
    if not equal:
        vals = struct.unpack("<%df" % n, got)
        refv = struct.unpack("<%df" % n, buf_ref)
        maxd = max(abs(x - y) for x, y in zip(vals, refv))
        bad = [i for i in range(n) if vals[i] != refv[i]]
        print("MISMATCH max_abs_diff=%.3e first_bad_idx=%s c[7]=%r ref[7]=%r"
              % (maxd, bad[:5], vals[7], refv[7]))
    else:
        vals = struct.unpack("<%df" % n, got)
        print("RESULT equal_bytes=True n=%d c[0]=%r c[7]=%r c[-1]=%r"
              % (n, vals[0], vals[7], vals[-1]))
    say("end")
    verdict = "PASS" if equal else "FAIL"
    print("PROBE151_VERDICT", verdict)
    return 0 if equal else 10


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "=== 跑 (c)：无 torch 编译 + ctypes 启动 ==="
timeout --foreground -k 5 240 "$PY_TRITON" -B -u "$T/151_launch.py"
RC=$?
echo "PY_RC=$RC"

echo "=== biir 落盘 ==="
ls -l "$T/biir.ll" 2>/dev/null || echo "NO_DUMP"

echo "=== 事后健康 ==="
if timeout --foreground -k 2 10 "$PY_VENDOR" -B -c \
    "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
    >/dev/null 2>&1; then
  echo "CARD_STILL_HEALTHY"
else
  echo "CARD_BROKEN_AFTER_PROBE"
fi

echo "PROBE151_DONE rc=$RC"
exit "$RC"
