set -u
# 探针 112：shim 的边界与多设备隔离（真机）。
#
# 为什么需要这一条：111 只验了"能分配、能往返、账本收敛"。但有几处一旦错就会静默出错，
# 而且都是 aten 迟早会踩到的形态：
#   1) **多设备**：沙箱里 device_count=2（物理 6、7）。池如果跨设备共用，0 号卡的块会被
#      发给 1 号卡上的张量——内核在错误的卡上访存，症状是"结果偶尔错"。本探针用
#      「0 号卡释放的块不许被 1 号卡复用」这条断言钉住它。
#   2) **零尺寸张量**：`torch.empty(0, 3)` 在 aten 里到处都是，不能向驱动申请 0 字节。
#   3) **0 维标量**：`torch.zeros([])`，存储 1 个元素。
#   4) **大块直通**（> 2 MiB 页）：走的代码路径与桶不同。
#   5) **init 幂等**：`register()` 被调两次不能重复注册分配器。
#
# 用法（投沙箱队列）：
#   python tools/remote.py sbx-run probes/112_shim_edges.sh --name 112_shim_edges --tail 0
# 期望 SHIM_EDGES_PASS / CARD_STILL_HEALTHY / PROBE112_DONE。

CX=/usr/local/corex
WORK=/data/home/2024240384035/coffin-work/base/sandbox/work
PY_SHARED=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PY28=/data/home/2024240384035/venvs/corex28/bin/python
SRC=$WORK/ct/src

export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡 ==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY_SHARED" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

echo
echo "=== 1. 边界与隔离 ==="
timeout --foreground -k 5 260 "$PY28" -B -u - "$SRC" <<'PYEOF'
"""shim 边界条件 + 多设备隔离。"""
import ctypes
import gc
import os
import sys

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

torch.utils.rename_privateuse1_backend("corex")
from corex_torch import _corex_shim as shim

shim.init(0)
DT = int(torch.float32) if hasattr(torch.float32, "__int__") else 6
print("device_count =", shim.device_count(), "| ScalarType(float32) =", DT)
ok = True

HOME = os.environ.get("COREX_HOME", "/usr/local/corex")
lib = ctypes.CDLL(os.path.join(HOME, "lib64", "libcudart.so.10.2"))
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
H2D, D2H = 1, 2


def check(label, cond):
    """打印一条断言结果。"""
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")
    ok = ok and cond


print("\n--- 1.1 空张量不向驱动申请（numel == 0）---")
shim.empty_cache()
r_before, a_before = shim.reserved_bytes(), shim.allocated_bytes()
z = shim.empty_strided([0, 3], [3, 1], DT, 0)
print("  shape =", tuple(z.shape), "| numel =", z.numel(), "| data_ptr =", z.data_ptr())
check("空张量不占账", shim.reserved_bytes() == r_before and shim.allocated_bytes() == a_before)
del z
gc.collect()

print("\n--- 1.2 0 维标量（存储 1 个元素 = 4 B）---")
s = shim.empty_strided([], [], DT, 0)
step = shim.allocated_bytes() - a_before
print("  shape =", tuple(s.shape), "| numel =", s.numel(), "| 记账增量 =", step, "B")
check("0 维标量记账 4 B", step == 4)
val = (ctypes.c_float * 1)(42.5)
lib.cudaMemcpy(ctypes.c_void_p(s.data_ptr()), ctypes.cast(val, ctypes.c_void_p), 4, H2D)
back = (ctypes.c_float * 1)()
lib.cudaMemcpy(ctypes.cast(back, ctypes.c_void_p), ctypes.c_void_p(s.data_ptr()), 4, D2H)
check("0 维标量可读写", abs(back[0] - 42.5) < 1e-6)
del s
gc.collect()

print("\n--- 1.3 大块直通（4 MiB > 2 MiB 页）---")
shim.empty_cache()
r0 = shim.reserved_bytes()
big = shim.empty_strided([1024 * 1024], [1], DT, 0)          # 4 MiB
r1 = shim.reserved_bytes()
print("  reserved 增量 =", r1 - r0, "B（按 2 MiB 页对齐，应 >= 4 MiB 且是 2 MiB 的整数倍）")
check("大块按页对齐", (r1 - r0) >= 4 * 1024 * 1024 and (r1 - r0) % (2 * 1024 * 1024) == 0)
hdr = (ctypes.c_float * 4)(7.0, 8.0, 9.0, 10.0)
lib.cudaMemcpy(ctypes.c_void_p(big.data_ptr()), ctypes.cast(hdr, ctypes.c_void_p), 16, H2D)
back = (ctypes.c_float * 4)()
lib.cudaMemcpy(ctypes.cast(back, ctypes.c_void_p), ctypes.c_void_p(big.data_ptr()), 16, D2H)
check("大块首部可读写", [round(v, 2) for v in back] == [7.0, 8.0, 9.0, 10.0])
del big
gc.collect()
shim.empty_cache()
check("大块 empty_cache 后归零", shim.reserved_bytes() == 0 and shim.allocated_bytes() == 0)

print("\n--- 1.4 多设备隔离：0 号卡释放的块不许发给 1 号卡 ---")
from corex_torch import _runtime as rt

if shim.device_count() < 2:
    print("  只有 1 台设备，跳过（本沙箱应有 2 台）")
else:
    rt.set_device(0)
    # 先在 0 号卡上分配再释放，让它的块留在 0 号卡的池里。
    t0 = shim.empty_strided([128], [1], DT, 0)              # 512 B，正好最小桶
    p0 = t0.data_ptr()
    print("  0 号卡首次分配 ptr = 0x%x | 当前 runtime 设备 = %d" % (p0, rt.current_device()))
    check("factory 之后 runtime 设备 = 0", rt.current_device() == 0)
    del t0
    gc.collect()
    saved0 = shim.reserved_bytes()
    print("  0 号卡池里已有缓存块，reserved =", saved0)
    # 现在到 1 号卡上要同样大小的一块。
    t1 = shim.empty_strided([128], [1], DT, 1)
    p1 = t1.data_ptr()
    print("  1 号卡分配 ptr = 0x%x device = %s | 当前 runtime 设备 = %d"
          % (p1, t1.device, rt.current_device()))
    check("1 号卡不复用 0 号卡的块", p1 != p0)
    check("张量设备号正确", str(t1.device) == "corex:1")
    check("factory 之后 runtime 设备 = 1（缓存池命中也要绑卡）", rt.current_device() == 1)
    check("1 号卡的块让 reserved 继续增长", shim.reserved_bytes() > saved0)
    # 回到 0 号卡要一块（缓存池命中），再分别写不同内容、分别读回。
    t0b = shim.empty_strided([128], [1], DT, 0)
    print("  0 号卡再次分配 ptr = 0x%x（应等于 0x%x = 池复用）| runtime 设备 = %d"
          % (t0b.data_ptr(), p0, rt.current_device()))
    check("0 号卡池复用的是同一块", t0b.data_ptr() == p0)
    check("factory 之后 runtime 设备切回 0", rt.current_device() == 0)

    def copy_h2d(ptr, host, nbytes, dev):
        """拷贝前显式切卡。**这一步不能省**：见下面 1.4b 的实测。"""
        rt.set_device(dev)
        return lib.cudaMemcpy(ctypes.c_void_p(ptr), ctypes.cast(host, ctypes.c_void_p),
                              nbytes, H2D)

    def copy_d2h(host, ptr, nbytes, dev):
        """读回前同样显式切卡。"""
        rt.set_device(dev)
        return lib.cudaMemcpy(ctypes.cast(host, ctypes.c_void_p), ctypes.c_void_p(ptr),
                              nbytes, D2H)

    # --- 1.4a 正确用法：每步 memcpy 前把 runtime 切到该指针所在的卡 ---
    v0 = (ctypes.c_float * 2)(1.0, 2.0)
    v1 = (ctypes.c_float * 2)(9.0, 9.5)
    b0 = (ctypes.c_float * 2)()
    b1 = (ctypes.c_float * 2)()
    rc_a = copy_h2d(t0b.data_ptr(), v0, 8, 0)
    rc_b = copy_h2d(t1.data_ptr(), v1, 8, 1)
    rc_c = copy_d2h(b0, t0b.data_ptr(), 8, 0)
    rc_d = copy_d2h(b1, t1.data_ptr(), 8, 1)
    print("  memcpy rc =", rc_a, rc_b, rc_c, rc_d,
          "| 0 号卡读回", [round(v, 2) for v in b0], "| 1 号卡读回", [round(v, 2) for v in b1])
    check("四步 memcpy 都 rc=0", rc_a == 0 and rc_b == 0 and rc_c == 0 and rc_d == 0)
    check("两卡数据互不干扰", [round(v, 2) for v in b0] == [1.0, 2.0]
          and [round(v, 2) for v in b1] == [9.0, 9.5])

    # --- 1.4b 反面实验：当前设备与指针不匹配时，这个 runtime **静默别名**、rc 还是 0 ---
    # 现象：0x394...（卡 1）与 0x304...（卡 0）低地址相同，卡 0 上当成同一块内存。
    # 结论：跨设备指针不许省 set_device，否则数据被悄悄写错，没有任何报错可查。
    rt.set_device(0)
    vbad = (ctypes.c_float * 2)(-7.0, -8.0)
    rc_bad = lib.cudaMemcpy(ctypes.c_void_p(t1.data_ptr()), ctypes.cast(vbad, ctypes.c_void_p),
                            8, H2D)                      # 故意不切卡
    rt.set_device(0)
    b0_after = (ctypes.c_float * 2)()
    lib.cudaMemcpy(ctypes.cast(b0_after, ctypes.c_void_p), ctypes.c_void_p(t0b.data_ptr()),
                   8, D2H)
    print("  反面实验 rc =", rc_bad, "| 本该写卡 1 的 -7/-8 之后，卡 0 上读回",
          [round(v, 2) for v in b0_after])
    alias = (rc_bad == 0 and [round(v, 2) for v in b0_after] == [-7.0, -8.0])
    print("  CROSS_DEVICE_SILENT_ALIAS" if alias else "  CROSS_DEVICE_SILENT_ALIAS_NOT_REPRODUCED")
    del t1, t0b
    gc.collect()

print("\n--- 1.5 init 幂等 ---")
before = shim.reserved_bytes()
shim.init(0)
shim.init(0)
check("重复 init 不重复注册/不炸", shim.is_initialized() and shim.reserved_bytes() == before)

print("\n--- 1.6 收尾：全部还给驱动 ---")
shim.empty_cache()
print("  reserved =", shim.reserved_bytes(), " allocated =", shim.allocated_bytes())
check("收尾归零", shim.reserved_bytes() == 0 and shim.allocated_bytes() == 0)
check("cuda.is_available 仍是 False", torch.cuda.is_available() is False)

print("\nSHIM_EDGES_PASS" if ok else "\nSHIM_EDGES_FAIL")
raise SystemExit(0 if ok else 1)
PYEOF
RC=$?
echo "SHIM_EDGES_RC=$RC"

echo
echo "=== 2. 收尾验卡 ==="
if timeout --foreground -k 2 20 "$PY_SHARED" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE112_DONE $(date -Is)"
exit $RC
