set -u
# 目的：设备层的全部能力最终都要落到 CoreX 的 runtime API 上（厂商 torch 2.1 自己也是这么干的）。
# 这份探针只问一件事：**不借用厂商 torch，只用 ctypes 打开 libcudart.so.10.2，
# 能不能把一个 context 用起来** —— 分配、H2D/D2H 往返、流、事件、属性查询。
# 它是 corex-torch 设备层能否成立的第一道门槛；driver API（libcuda.so.1）那条路由
# 另一个探针同时验，两边都通就选更省事的那个。
#
# 为什么用 runtime 而不是 driver：runtime API 会自己延迟建 primary context，
# 省掉 cuInit/cuCtxCreate 那一步（后者在本平台坏卡上挂死过，见 corex-triton/docs/00-FACTS.md §5）。
CX=/usr/local/corex
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 先验卡（坏卡直接退出，不占队列）==="
# 注意：沙箱 runner 已经把 HEALTHY_IDS 写进 CUDA_VISIBLE_DEVICES（形如 6,7），子进程里
# 逻辑 0 号就是物理 6 号。**在这里再写 CUDA_VISIBLE_DEVICES=0 会覆盖成物理 0 号（坏卡）**
# —— 那是 2026-11-06 踩过的坑：探针一上来就 CARD_UNHEALTHY。只在外部没设时才逐卡挑。
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if ! timeout --foreground -k 2 20 "$PY" -B -c \
            "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
            >/dev/null 2>&1; then
        echo "CARD_UNHEALTHY=exit9"
        exit 9
    fi
else
    PICKED=none
    for dev in 0 1 2 3 4 5 6 7; do
        if CUDA_VISIBLE_DEVICES=$dev timeout --foreground -k 2 10 "$PY" -B -c \
                "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
                >/dev/null 2>&1; then
            PICKED=$dev
            break
        fi
    done
    if [ "$PICKED" = none ]; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
    export CUDA_VISIBLE_DEVICES=$PICKED
    echo "PICKED_PHYSICAL=$PICKED"
fi
echo "CARD_OK"

echo "=== 1. 用 ctypes 打开 libcudart 并查符号 ==="
timeout --foreground -k 5 120 "$PY" -B -u - "$CX" <<'PYEOF'
import ctypes, sys, os
cx = sys.argv[1]
path = os.path.join(cx, "lib64", "libcudart.so.10.2")
rt = ctypes.CDLL(path)
print("loaded:", path)

need = ["cudaGetDeviceCount", "cudaSetDevice", "cudaGetDevice", "cudaGetDeviceProperties",
        "cudaDeviceSynchronize", "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync",
        "cudaMemset", "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize",
        "cudaStreamQuery", "cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize",
        "cudaEventElapsedTime", "cudaGetLastError", "cudaPeekAtLastError", "cudaGetErrorString",
        "cudaDeviceGetAttribute", "cudaMemGetInfo", "cudaHostAlloc", "cudaDeviceReset",
        "cudaMemcpyKind_dummy"]
missing = []
for name in need:
    try:
        getattr(rt, name)
    except AttributeError:
        missing.append(name)
print("missing symbols:", missing or "(none)")

# 版本三件套：runtime 自报的版本号，用来判断这层到底实现了哪一代 API
class _Ver(ctypes.Structure):
    _fields_ = [("major", ctypes.c_int), ("minor", ctypes.c_int), ("patch", ctypes.c_int)]
v = _Ver()
for fn in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
    try:
        getattr(rt, fn)(ctypes.byref(v))
        print(f"{fn}: {v.major}.{v.minor}.{v.patch}")
    except Exception as e:                                  # noqa: BLE001
        print(f"{fn}: 调用失败 {e}")

print("=== 2. context 能否由 runtime 自己立起来 ===")
print("cudaGetDeviceCount rc =", rt.cudaGetDeviceCount(ctypes.byref(ctypes.c_int())))
print("cudaSetDevice(0) rc  =", rt.cudaSetDevice(0))
dev = ctypes.c_int(-1)
print("cudaGetDevice rc =", rt.cudaGetDevice(ctypes.byref(dev)), "dev =", dev.value)

class _Props(ctypes.Structure):
    # cudaDeviceProp 在本平台是 CUDA 10.2 的布局；这里只取前若干个字段，
    # 用 512 字节的原始 buffer 兜住，避免结构体对不上（只读 name 之外的属性用 attribute 查）。
    _fields_ = [("name", ctypes.c_char * 256), ("_rest", ctypes.c_byte * 4096)]

props = _Props()
rc = rt.cudaGetDeviceProperties(ctypes.byref(props), 0)
print("cudaGetDeviceProperties rc =", rc, "name =", props.name.decode(errors="replace"))

def attr(a):
    val = ctypes.c_int(0)
    rc = rt.cudaDeviceGetAttribute(ctypes.byref(val), a, 0)
    return (rc, val.value)

# 这几个 attribute id 是 CUDA 里长期稳定的：1=maxThreadsPerBlock 2=maxBlockDimX
# 10=warpSize 16=multiProcessorCount 81=sharedMemPerMultiprocessor 8=sharedMemPerBlock
for name, aid in (("maxThreadsPerBlock", 1), ("warpSize", 10), ("multiProcessorCount", 16),
                  ("sharedMemPerBlock", 8), ("globalL1CacheSupported", 93)):
    print(f"  attr[{name}] =", attr(aid))

free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
print("cudaMemGetInfo rc =", rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
      "free = %.2f GiB total = %.2f GiB" % (free.value / 2**30, total.value / 2**30))

print("=== 3. 分配 + H2D/D2H 逐字节往返 ===")
rt.cudaMalloc.restype = ctypes.c_int
rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
buf = ctypes.c_void_p()
rc = rt.cudaMalloc(ctypes.byref(buf), 1 << 20)
print("cudaMalloc(1 MiB) rc =", rc, "ptr = 0x%x" % (buf.value or 0))

src = bytes(range(256)) * 4096            # 1 MiB，内容有区分度
h_in = ctypes.create_string_buffer(src, len(src))
rc = rt.cudaMemcpy(ctypes.c_void_p(buf.value), h_in, len(src), 1)   # 1 = H2D
print("cudaMemcpy H2D rc =", rc)
h_out = ctypes.create_string_buffer(len(src))
rc = rt.cudaMemcpy(h_out, ctypes.c_void_p(buf.value), len(src), 2)  # 2 = D2H
print("cudaMemcpy D2H rc =", rc)
same = bytes(h_out.raw) == src
print("ROUNDTRIP_BYTES_EQUAL =", same)
if not same:
    diff = next(i for i, (a, b) in enumerate(zip(h_out.raw, src)) if a != b)
    print("  首个不一致偏移 =", diff, "读回", h_out.raw[diff], "期望", src[diff])

print("=== 4. 流与事件 ===")
rt.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
stream = ctypes.c_void_p()
print("cudaStreamCreate rc =", rt.cudaStreamCreate(ctypes.byref(stream)), "stream = 0x%x" % (stream.value or 0))
rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
print("cudaMemcpyAsync rc =", rt.cudaMemcpyAsync(ctypes.c_void_p(buf.value), h_in, len(src), 1, stream))
print("cudaStreamSynchronize rc =", rt.cudaStreamSynchronize(stream))

ev0, ev1 = ctypes.c_void_p(), ctypes.c_void_p()
print("cudaEventCreate rc =", rt.cudaEventCreate(ctypes.byref(ev0)))
rt.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
print("cudaEventRecord rc =", rt.cudaEventRecord(ev0, stream))
print("cudaEventSynchronize rc =", rt.cudaEventSynchronize(ev0))
ms = ctypes.c_float(0)
rt.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
print("cudaEventElapsedTime rc =", rt.cudaEventElapsedTime(ctypes.byref(ms), ev0, ev0), "ms =", ms.value)

print("=== 5. 错误码语义 ===")
print("cudaPeekAtLastError =", rt.cudaPeekAtLastError())
rt.cudaGetErrorString.restype = ctypes.c_char_p
print("cudaGetErrorString(0) =", rt.cudaGetErrorString(0))
print("cudaGetErrorString(1) =", rt.cudaGetErrorString(1))
print("cudaDeviceSynchronize rc =", rt.cudaDeviceSynchronize())
print("RT_DONE")
PYEOF
echo "RC=$?"
