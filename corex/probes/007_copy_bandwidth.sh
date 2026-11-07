set -u
# 目的：给设备层的「拷贝」定口径。这件事的优先级很高，因为**厂商 torch 2.1 的 copy_ 是驱动 DMA
# 引擎做的（471 GB/s），不是 kernel**（docs/00-FACTS.md §9）。我们的设备后端如果拿 Triton 逐元素
# 内核算拷贝，会直接丢掉 1.6 倍带宽，还白占 SM。所以先用我们自己的绑定量三件事：
#   1. D2D 大块拷贝的实际带宽（决定 copy_ 能不能追平厂商那条线）；
#   2. H2D/D2H 在 pageable 与 pinned 两种主机内存下的差异（驱动审计说 host 路径每次都要
#      get_user_pages_fast 重新 pin，那是毫秒级——这里量出它到底有多贵）；
#   3. 小拷贝的固定开销（torch 里 .to()/.cpu() 常常只有几 KB）。
# 这一段全程只用 ctypes + libcudart，不 import 任何 torch。
CX=/usr/local/corex
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
SRC=/data/home/2024240384035/coffin-work/base/sandbox/work/ct/src
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
echo "HOST=$(hostname) DATE=$(date -Is)"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

cat > /data/home/2024240384035/coffin-work/base/sandbox/work/ct/bench_copy.py <<'PYEOF'
import ctypes
import sys
import time

sys.path.insert(0, sys.argv[1])
from corex_torch import _runtime as rt

rt.set_device(0)
stream = rt.Stream.create()
print("device =", rt.device_name(0), " warp =", rt.device_attribute(rt.ATTR_WARP_SIZE))

def timed(fn, reps):
    """跑 reps 次并同步，返回 (平均 ms, 第一次的 ms)。第一次代表冷启动（可能含首次 pin）。"""
    first = None
    t0 = time.perf_counter()
    for i in range(reps):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if i == 0:
            first = dt
    rt.synchronize()
    total = time.perf_counter() - t0
    return 1000 * total / reps, 1000 * first

print("=== 1. D2D 拷贝带宽（cudaMemcpyAsync，走 mem_handle，不 pin 页）===")
for mib in (1, 16, 64, 256):
    n = mib << 20
    a, b = rt.malloc(n), rt.malloc(n)
    reps = max(3, min(50, (1 << 30) // n))
    avg, first = timed(lambda: rt.memcpy(a, b, n, rt.MEMCPY_DEVICE_TO_DEVICE, stream), reps)
    gbs = 2 * n / (avg / 1000) / 1e9       # 读 + 写都算流量
    print(f"  {mib:4d} MiB  avg {avg:8.3f} ms  first {first:7.3f} ms  -> {gbs:7.1f} GB/s")
    rt.free(a); rt.free(b)

print("=== 2. H2D / D2H：pageable vs pinned ===")
for mib in (1, 16, 64):
    n = mib << 20
    dev = rt.malloc(n)
    pageable = ctypes.create_string_buffer(n)          # 普通内存，驱动每次都要重新 pin
    pinned = rt.malloc_host(n)                          # 页锁定，省掉 get_user_pages_fast
    lo = ctypes.addressof(pageable)
    for label, host in (("pageable", lo), ("pinned", pinned.value)):
        avg, first = timed(lambda: rt.memcpy(dev, host, n, rt.MEMCPY_HOST_TO_DEVICE, stream), 5)
        gbs = n / (avg / 1000) / 1e9
        print(f"  {mib:3d} MiB H2D {label:9s} avg {avg:8.3f} ms  first {first:7.3f} ms -> {gbs:6.1f} GB/s")
    rt.free_host(pinned)
    rt.free(dev)

print("=== 3. 小拷贝的固定开销（torch 的 .to()/.cpu() 常常只有几十 KB）===")
for kb in (1, 64, 1024):
    n = kb << 10
    dev = rt.malloc(n)
    host = ctypes.create_string_buffer(n)
    pinned = rt.malloc_host(n)
    for label, hp in (("pageable", ctypes.addressof(host)), ("pinned", pinned.value)):
        avg, first = timed(lambda: rt.memcpy(dev, hp, n, rt.MEMCPY_HOST_TO_DEVICE, stream), 200)
        print(f"  {kb:5d} KiB H2D {label:9s} avg {avg * 1000:8.2f} us  first {first * 1000:9.2f} us")
    rt.free_host(pinned)
    rt.free(dev)

print("=== 4. 纯流同步的开销（给「一步里能插多少次同步」定预算）===")
avg, _ = timed(lambda: rt.Stream.synchronize(stream), 200)
print(f"  cudaStreamSynchronize avg {avg * 1000:7.2f} us")
avg, _ = timed(lambda: rt.synchronize(), 200)
print(f"  cudaDeviceSynchronize avg {avg * 1000:7.2f} us")

stream.close()
print("COPY_BENCH_DONE")
PYEOF

timeout --foreground -k 5 240 "$PY" -B -u /data/home/2024240384035/coffin-work/base/sandbox/work/ct/bench_copy.py "$SRC"
PYRC=$?
echo "RC=$PYRC"
if timeout --foreground -k 2 20 "$PY" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi
exit $PYRC
