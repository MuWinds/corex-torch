set -u
# 探针 111：沙箱真机验收 corex_shim —— 分配器、strided 张量、D2H 校验、缓存池收敛（验收 B）。
#
# 为什么这么设计：
#   1. **全程不碰 torch 的算子**：这一刻 Python 侧还只注册了设备改名，`aten::empty` 之类都还没有
#      实现（那是 _device.py 的事）。所以校验值必须靠 `cudaMemcpy` D2H 拿回来，
#      不能指望 `t.cpu()` / `print(t)`——后者会走 `aten::_copy_from` / `aten::view` 直接报未实现。
#   2. **要验"分配器真的在管显存"**：只看 `torch.empty` 成功不够（探针 102 用 host malloc 也能成功）。
#      所以三段账本都要对：借出值、缓存值、`empty_cache()` 之后归零。
#   3. **要验非连续 strides 原样生效**：否则后面 `view`/`transpose` 会静默算错。
#   4. **顺手取回"下一步要补哪些算子"**：把 `torch.zeros/+/sum/.cpu()` 此刻的报错原文抓下来，
#      那就是 _device.py 的算子清单。
#   5. 卡是共享且不可恢复的（`recover=0`），所以先验卡、坏卡 exit 9、结尾再验一次。
#
# 前置：110 号探针已经编好 `$WORK/ct/src/corex_torch/_corex_shim.so`。
# 用法（投沙箱队列，不要直接跑）：
#   python tools/remote.py sbx-run probes/111_shim_smoke.sh --name 111_shim_smoke --tail 0
# 期望 CARD_OK → SHIM_SMOKE_PASS / SHIM_ALLOC_PASS → CARD_STILL_HEALTHY → PROBE111_DONE。

CX=/usr/local/corex
WORK=/data/home/2024240384035/coffin-work/base/sandbox/work
#: 厂商 torch（共享 py310）：只用来验卡，不用它跑我们的后端
PY_SHARED=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
#: torch 2.8 隔离 venv：shim 必须在这个环境里跑
PY28=/data/home/2024240384035/venvs/corex28/bin/python
SRC=$WORK/ct/src

export LD_LIBRARY_PATH=$CX/lib64:$CX/lib:${LD_LIBRARY_PATH:-}
echo "HOST=$(hostname) DATE=$(date -Is)"
echo "INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== 0. 验卡（继承 runner 的 CUDA_VISIBLE_DEVICES，不自己设）==="
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "NO_INHERITED_CVD=exit9"; exit 9; fi
if ! timeout --foreground -k 2 20 "$PY_SHARED" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_UNHEALTHY=exit9"; exit 9; fi
echo "CARD_OK"

if [ ! -f "$SRC/corex_torch/_corex_shim.so" ]; then
    echo "SHIM_SO_MISSING=$SRC/corex_torch/_corex_shim.so（先跑 probes/110_shim_build.sh）"
    exit 3
fi

echo
echo "=== 1. 真机跑 shim ==="
timeout --foreground -k 5 260 "$PY28" -B -u - "$SRC" <<'PYEOF'
"""真机验收脚本（由 111_shim_smoke.sh 通过 heredoc 喂给 python）。"""
import ctypes
import gc
import os
import re
import sys
import traceback

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import torch

print("torch", torch.__version__, "| cuda.is_available =", torch.cuda.is_available())
torch.utils.rename_privateuse1_backend("corex")
print("privateuse1 backend name =", torch._C._get_privateuse1_backend_name())

from corex_torch import _corex_shim as shim

print("shim 导入 OK；未 init 时 is_initialized =", shim.is_initialized())
shim.init(0)
print("init(0) 之后 is_initialized =", shim.is_initialized(),
      "| device_count =", shim.device_count())

# torch 的 factory 函数会先去找 `torch.corex` 这个 device module（C++ 的 device_lazy_init
# 里 `import torch.<name>`），找不到就报 `ModuleNotFoundError: No module named 'torch.corex'`，
# 根本走不到 dispatch。_device.py 以后会提供真的那个；这里先挂一个桩，好让第 6 段能问到
# 「缺哪些算子」——这正是本探针要交给 _device.py 的清单。
class _StubDeviceModule:
    """最小 device module 桩（接口清单见 docs/03-TORCH28-API.md §2.4）。"""

    is_available = staticmethod(lambda: True)
    device_count = staticmethod(lambda: shim.device_count())
    current_device = staticmethod(lambda: 0)
    _is_in_bad_fork = staticmethod(lambda: False)
    manual_seed_all = staticmethod(lambda seed: None)
    _lazy_init = staticmethod(lambda: None)

    @staticmethod
    def get_rng_state(device=None):
        return torch.zeros(1, dtype=torch.uint8)

    @staticmethod
    def set_rng_state(state, device=None):
        return None

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float32]


torch._register_device_module("corex", _StubDeviceModule)
print("已挂 device module 桩：hasattr(torch,'corex') =", hasattr(torch, "corex"))

# dtype 的整数值 = c10::ScalarType。先试 int(torch.dtype)，不行再查表——
# 这个值 _device.py 也要用，所以把实测结果打出来。
def scalar_type(dt):
    """torch.dtype → c10::ScalarType 的整数。"""
    try:
        return int(dt)
    except Exception:                                  # noqa: BLE001 - 老版本没有 __int__
        table = {torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3,
                 torch.int64: 4, torch.float16: 5, torch.float32: 6, torch.float64: 7,
                 torch.bool: 11, torch.bfloat16: 15}
        return table[dt]

DT = scalar_type(torch.float32)
print("ScalarType(float32) =", DT, "（int(torch.float32) 可用）" if DT == 6 else "（走了查表）")

# libcudart 只用 ctypes 来搬数据；argtypes 写全，否则 60 位设备指针会被截断。
HOME = os.environ.get("COREX_HOME", "/usr/local/corex")
lib = ctypes.CDLL(os.path.join(HOME, "lib64", "libcudart.so.10.2"))
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
lib.cudaDeviceSynchronize.argtypes = []
lib.cudaDeviceSynchronize.restype = ctypes.c_int
H2D, D2H = 1, 2

ok = True

print("\n=== 2. empty_strided：连续张量 + H2D/D2H 往返 ===")
t = shim.empty_strided([2, 3], [3, 1], DT, 0)
print("  device =", t.device, "| dtype =", t.dtype, "| shape =", tuple(t.shape),
      "| stride =", tuple(t.stride()), "| numel =", t.numel(),
      "| contiguous =", t.is_contiguous())
print("  data_ptr = 0x%x" % t.data_ptr())
host = (ctypes.c_float * 6)(1.5, -2.5, 3.25, 4.75, 5.5, 6.0)
rc = lib.cudaMemcpy(ctypes.c_void_p(t.data_ptr()), ctypes.cast(host, ctypes.c_void_p),
                    24, H2D)
print("  H2D rc =", rc)
back = (ctypes.c_float * 6)()
rc = lib.cudaMemcpy(ctypes.cast(back, ctypes.c_void_p), ctypes.c_void_p(t.data_ptr()),
                    24, D2H)
print("  D2H rc =", rc, "读回 =", [round(v, 3) for v in back])
got = [round(v, 3) for v in back]
want = [1.5, -2.5, 3.25, 4.75, 5.5, 6.0]
if got != want:
    print("  ROUNDTRIP_MISMATCH"); ok = False
else:
    print("  ROUNDTRIP_MATCH")

print("\n=== 3. 非连续 strides 必须原样生效（否则 view/transpose 会静默算错）===")
t2 = shim.empty_strided([2, 3], [1, 2], DT, 0)      # 转置后的布局
print("  shape =", tuple(t2.shape), "| stride =", tuple(t2.stride()),
      "| contiguous =", t2.is_contiguous())
if tuple(t2.stride()) != (1, 2) or t2.is_contiguous():
    print("  STRIDE_PASSTHROUGH_BAD"); ok = False
else:
    print("  STRIDE_PASSTHROUGH_OK")
# 同一块存储按 [1,2] 写入 6 个值，再按同样 stride 读回，验证 strides 不是"摆设"。
vals = (ctypes.c_float * 6)(10, 20, 30, 40, 50, 60)
lib.cudaMemcpy(ctypes.c_void_p(t2.data_ptr()), ctypes.cast(vals, ctypes.c_void_p), 24, H2D)
readback = (ctypes.c_float * 6)()
lib.cudaMemcpy(ctypes.cast(readback, ctypes.c_void_p), ctypes.c_void_p(t2.data_ptr()), 24, D2H)
print("  存储顺序读回 =", [int(v) for v in readback])
if [int(v) for v in readback] != [10, 20, 30, 40, 50, 60]:
    print("  STRIDED_STORAGE_BAD"); ok = False

print("\n=== 4. from_blob 不接管所有权（同一指针先包装、再自己释放）===")
from corex_torch import _runtime as rt
rt.set_device(0)
raw = rt.malloc(64)
tb = shim.from_blob(raw.value, [4, 4], [4, 1], DT, 0)
print("  from_blob OK: shape =", tuple(tb.shape), "| stride =", tuple(tb.stride()),
      "| device =", tb.device)
src = (ctypes.c_float * 16)(*range(16))
lib.cudaMemcpy(ctypes.c_void_p(tb.data_ptr()), ctypes.cast(src, ctypes.c_void_p), 64, H2D)
back2 = (ctypes.c_float * 16)()
lib.cudaMemcpy(ctypes.cast(back2, ctypes.c_void_p), ctypes.c_void_p(tb.data_ptr()), 64, D2H)
print("  经 from_blob 张量写入并读回 =", [int(v) for v in back2][:6], "...")
del tb
gc.collect()
try:
    rt.free(raw)          # 若 shim 抢了所有权，这里会二次释放/报错
    print("  FROM_BLOB_NO_OWNERSHIP_OK（包装张量析构后仍能自己 free）")
except Exception as exc:                                # noqa: BLE001
    print("  FROM_BLOB_NO_OWNERSHIP_BAD:", exc); ok = False

print("\n=== 5. 分配器：200 次不同尺寸申请 → 全释放 → empty_cache 收敛 ===")
# 先清基线：前面章节建的张量还活着，它们占的账必须扣掉，否则「归零」这个判据不成立
# （第一版探针就是因为没清基线，把 t/t2 的 36 B 误报成泄漏）。
del t, t2
gc.collect()
shim.empty_cache()
r0, a0 = shim.reserved_bytes(), shim.allocated_bytes()
print("  基线（前面的张量都释放 + empty_cache）: reserved =", r0, " allocated =", a0)
if r0 != 0 or a0 != 0:
    print("  BASELINE_NOT_CLEAN_BAD"); ok = False

# 回归：连续 [2,3] 张量的存储量必须是 6*4=24 B。曾经因为「取各维最大跨度」算错成 16 B，
# 多维张量会写越界；这条断言就是钉住那个 bug 的。
probe_t = shim.empty_strided([2, 3], [3, 1], DT, 0)
step = shim.allocated_bytes() - a0
print("  [2,3] 连续张量应记账 24 B，实测", step, "B")
if step != 24:
    print("  STORAGE_BYTES_BAD"); ok = False
else:
    print("  STORAGE_BYTES_OK")
del probe_t
gc.collect()
shim.empty_cache()

small = [4, 12, 36, 100, 260, 600, 1500, 4000, 9000, 40000]      # 覆盖多个桶
keep = []
for i in range(200):
    n = small[i % len(small)]
    keep.append(shim.empty_strided([n], [1], DT, 0))
r1, a1 = shim.reserved_bytes(), shim.allocated_bytes()
print("  200 个张量活着: reserved =", r1, " allocated =", a1,
      " slab 对齐 =", shim.slab_alignment())
if a1 <= 0 or r1 <= 0:
    print("  ALLOC_NOT_TRACKED_BAD"); ok = False
del keep
gc.collect()
r2, a2 = shim.reserved_bytes(), shim.allocated_bytes()
print("  全部释放后  : reserved =", r2, " allocated =", a2,
      "（reserved 应等于", r1, "= 池还在缓存；allocated 应回 0）")
if a2 != 0:
    print("  ALLOC_NOT_RETURNED_BAD"); ok = False
shim.empty_cache()
r3, a3 = shim.reserved_bytes(), shim.allocated_bytes()
print("  empty_cache : reserved =", r3, " allocated =", a3, "（应双双归零）")
alloc_pass = (a0 == 0 and r0 == 0 and a1 > 0 and r1 > 0 and a2 == 0 and r2 == r1
              and r3 == 0 and a3 == 0)
print("  SHIM_ALLOC_PASS" if alloc_pass else "  SHIM_ALLOC_FAIL")
ok = ok and alloc_pass

print("\n=== 6. 现在 torch.zeros(4, device='corex') 报什么（= 下一步要补的算子清单）===")
def first_line(exc):
    """取异常消息首行，并把里面的 aten 算子名抓出来。"""
    text = str(exc)
    head = text.strip().splitlines()[0] if text.strip() else ""
    ops = sorted(set(re.findall(r"aten::[A-Za-z0-9_.]+", text)))
    return head[:220], ops

tries = [("torch.zeros(4, device='corex')", lambda: torch.zeros(4, device="corex")),
         ("torch.empty(4, device='corex')", lambda: torch.empty(4, device="corex")),
         ("torch.tensor([1.], device='corex')", lambda: torch.tensor([1.0], device="corex"))]
for label, fn in tries:
    try:
        value = fn()
        print(f"  [{label}] OK -> {value.device}")
    except BaseException as exc:                        # noqa: BLE001
        head, ops = first_line(exc)
        print(f"  [{label}] {type(exc).__name__}: {head}")
        print(f"      涉及算子: {ops}")

# 把 empty.memory_format 用 shim 顶上去，再看下一跳卡在哪（这才是真正的算子清单）
def contig_strides(size):
    """行主序连续布局的 strides（shim 要求 strides 原样传入，不给默认值）。"""
    strides = []
    acc = 1
    for dim in reversed(list(size)):
        strides.append(acc)
        acc *= max(1, dim)
    return list(reversed(strides))


lib_aten = torch.library.Library("aten", "IMPL")
lib_aten.impl("empty.memory_format",
              lambda size, *, dtype=None, layout=None, device=None, pin_memory=None,
              memory_format=None: shim.empty_strided(
                  list(size), contig_strides(size),
                  scalar_type(torch.float32 if dtype is None else dtype), 0),
              "PrivateUse1")
print("\n  （已用 shim 挂上 aten::empty.memory_format，继续看下一跳）")
for label, fn in [("torch.zeros(4, device='corex')", lambda: torch.zeros(4, device="corex")),
                  ("t + 1", lambda: shim.empty_strided([4], [1], DT, 0) + 1),
                  ("t.sum()", lambda: shim.empty_strided([4], [1], DT, 0).sum()),
                  ("t.to('cpu')", lambda: shim.empty_strided([4], [1], DT, 0).to("cpu"))]:
    try:
        value = fn()
        print(f"  [{label}] OK -> {value.device}")
    except BaseException as exc:                        # noqa: BLE001
        head, ops = first_line(exc)
        print(f"  [{label}] {type(exc).__name__}: {head}")
        print(f"      涉及算子: {ops}")

print("\n  补充事实：backend 名 =", torch._C._get_privateuse1_backend_name(),
      "| cuda.is_available =", torch.cuda.is_available(),
      "（必须仍是 False）")
if torch.cuda.is_available():
    ok = False

print("\nSHIM_SMOKE_PASS" if ok else "\nSHIM_SMOKE_FAIL")
raise SystemExit(0 if ok else 1)
PYEOF
RC=$?
echo "SHIM_TEST_RC=$RC"

echo
echo "=== 7. 收尾验卡（R10：卡挂了要立刻知道）==="
if timeout --foreground -k 2 20 "$PY_SHARED" -B -c \
        "import torch;a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda');b=a@a;torch.cuda.synchronize();raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then echo "CARD_STILL_HEALTHY"; else echo "CARD_DEAD_AFTER"; fi

echo "PROBE111_DONE $(date -Is)"
exit $RC
