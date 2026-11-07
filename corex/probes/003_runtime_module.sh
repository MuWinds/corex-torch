set -u
# 目的：验收 corex-torch 自己的设备层（src/corex_torch/_runtime.py），而不是再写一遍 ctypes 探针。
# 它要回答三件事：
#   1. 我们的封装在真机上能不能拿到设备属性、分配、往返拷贝；——设备层成立的前提
#   2. 事件能不能「同时挂起多个」——docs/corex_drive/GRAPH-CHANNEL.md 说每个队列只有 4 个 user fence 槽，
#      这决定了 torch 侧 Event 的实现能不能随手创建（docs/05-DRIVER-CONSTRAINTS.md R3）；
#   3. cudaMalloc / cudaEventRecord 的固定开销——torch 会频繁申请小块，分配器要不要做缓存池。
# 编译/算子还没接，这一段刻意只测设备层。
CX=/usr/local/corex
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
SRC=/data/home/2024240384035/coffin-work/base/sandbox/work/ct/src
export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
echo "HOST=$(hostname) DATE=$(date -Is) SRC=$SRC"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES，不覆盖）==="
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if ! timeout --foreground -k 2 20 "$PY" -B -c \
            "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
            >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
else
    echo "NO_INHERITED_CVD=exit9"; exit 9
fi
echo "CARD_OK"

echo "=== 1. 设备属性（走 corex_torch._runtime）==="
timeout --foreground -k 5 180 "$PY" -B -u - "$SRC" <<'PYEOF'
import sys, os, time
sys.path.insert(0, sys.argv[1])
import corex_torch
from corex_torch import _runtime as rt

print("corex_torch", corex_torch.__version__, "avail =", corex_torch.is_available())
print("runtime_version =", rt.runtime_version(), "device_count =", rt.device_count())
for k, v in rt.properties(0).items():
    if isinstance(v, int) and v > 1 << 20:
        print(f"  {k} = {v} ({v / 2**30:.2f} GiB)")
    else:
        print(f"  {k} = {v}")

print("=== 2. 分配 + 逐字节往返（用我们自己的封装）===")
n = 1 << 20
ptr = rt.malloc(n)
payload = bytes((i * 7 + 13) & 0xFF for i in range(256)) * (n // 256)
import ctypes
src_buf = ctypes.create_string_buffer(payload, n)
rt.memcpy(ptr, ctypes.cast(src_buf, ctypes.c_void_p), n, rt.MEMCPY_HOST_TO_DEVICE)
out = ctypes.create_string_buffer(n)
rt.memcpy(out, ptr, n, rt.MEMCPY_DEVICE_TO_HOST)
print("ROUNDTRIP_BYTES_EQUAL =", out.raw == payload)
rt.free(ptr)

print("=== 3. 事件同时挂起数（fence 槽压力）===")
stream = rt.Stream.create()
ok, fail = 0, 0
events = []
for i in range(16):
    try:
        ev = rt.Event()
        ev.record(stream)
        events.append(ev)
        ok += 1
    except Exception as exc:                                # noqa: BLE001
        fail += 1
        print(f"  第 {i} 个事件创建/记录失败：{exc}")
print(f"  16 个事件：成功 {ok} 失败 {fail}")
for i, ev in enumerate(events):
    try:
        ev.synchronize()
    except Exception as exc:                                # noqa: BLE001
        print(f"  第 {i} 个事件等待失败：{exc}")
print("  全部 synchronize 完成")
ev0, ev1 = rt.Event(), rt.Event()
ev0.record(stream)
ev1.record(stream)
ev1.synchronize()
print("  一对事件的 elapsed_ms =", ev0.elapsed_ms(ev1))
for ev in events + [ev0, ev1]:
    ev.close()
stream.close()

print("=== 4. 分配与记录的固定开销（决定分配器要不要缓存）===")
for nbytes in (1024, 1 << 20, 64 << 20):
    t0 = time.perf_counter()
    reps = 50 if nbytes < (1 << 24) else 10
    ptrs = [rt.malloc(nbytes) for _ in range(reps)]
    t1 = time.perf_counter()
    for p in ptrs:
        rt.free(p)
    t2 = time.perf_counter()
    print(f"  {nbytes:>10} B: malloc {(t1 - t0) / reps * 1e6:8.1f} us/次  free {(t2 - t1) / reps * 1e6:8.1f} us/次")

s = rt.Stream.create()
ev_a, ev_b = rt.Event(), rt.Event()
t0 = time.perf_counter()
for _ in range(200):
    ev_a.record(s)
    ev_b.record(s)
    ev_b.synchronize()
t1 = time.perf_counter()
print(f"  record+record+sync 一轮 = {(t1 - t0) / 200 * 1e6:8.1f} us")
ev_a.close(); ev_b.close(); s.close()

print("=== 5. 空往返（同步语义是否真的等到设备） ===")
ptr = rt.malloc(4096)
t0 = time.perf_counter()
for _ in range(200):
    rt.synchronize()
t1 = time.perf_counter()
print(f"  cudaDeviceSynchronize = {(t1 - t0) / 200 * 1e6:8.1f} us/次")
rt.free(ptr)
print("RUNTIME_MODULE_DONE")
PYEOF
echo "RC=$?"
