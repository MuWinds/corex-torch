set -u
# 探针 107：厂商 torch 2.1 与上游 torch 2.8 在「设备后端 / 流 / 事件 / 分配器」上的接口差异。
#
# 为什么做这个探测：corex-torch 要在 2.8 上重做一遍 2.1 已经做过的事，而 2.1 那份是厂商
# 改过的（torch-2.1.0+corex.3.2.1）。两边差在哪儿必须逐条看源码，不然会照抄 2.1 的写法
# 然后在 2.8 上编不过（本探针已经因此发现两处：Allocator::copy_data 是新纯虚、
# allocate 掉了 const）。
#
# 只读：厂商环境（共享 py310）一个字节都不动，只用 diff / grep 读它的 site-packages。
# 注意：登录节点 **不能** import 厂商 torch（缺 libcufft.so.10），所以 2.1 侧只有源码证据，
# 运行时行为只能是 UNKNOWN。
#
# 用法（登录节点直接跑）：
#   python tools/remote.py put probes/107_torch21_vs_28_diff.sh /data/home/2024240384035/p107.sh
#   python tools/remote.py sh 'bash /data/home/2024240384035/p107.sh > /data/home/2024240384035/p107.out 2>&1'

set -e
V21=/data/software/Anaconda3/2024.10-1/envs/py310/lib/python3.10/site-packages/torch
V28=/data/home/2024240384035/venvs/corex28/lib/python3.10/site-packages/torch

echo "=== 0. 两边的版本号 ==="
echo "--- 2.1（厂商）---"; head -5 "$V21/version.py"
echo "--- 2.8（上游 CPU wheel）---"; head -5 "$V28/version.py"
echo "--- 2.1 是不是厂商改过的：看 _C 里有没有 corex 字样 ---"
grep -ac "corex" "$V21/lib/libtorch_python.so" 2>/dev/null || echo "0"
grep -ac "privateuse1" "$V21/lib/libtorch_python.so" 2>/dev/null || echo "0"

echo
echo "=== 1. Python 文件存在性（2.8 新增的模块/文件） ==="
for f in accelerator/__init__.py _library/__init__.py utils/_device.py _dynamo/device_interface.py; do
    p21=no; p28=no
    test -e "$V21/$f" && p21=yes
    test -e "$V28/$f" && p28=yes
    echo "  $f: 2.1=$p21 2.8=$p28"
done

echo
echo "=== 2. backend_registration.py 差异（Python 侧设备注册契约） ==="
diff -u "$V21/utils/backend_registration.py" "$V28/utils/backend_registration.py" | head -70 || true

echo
echo "=== 3. torch/__init__.py 里的模块发现机制 ==="
echo "--- 2.1 ---"
grep -n "_register_device_module\|_get_device_module\|import_module" "$V21/__init__.py" | head -10
echo "--- 2.8 ---"
grep -n "_register_device_module\|_get_device_module\|import_module" "$V28/__init__.py" | head -10

echo
echo "=== 4. c10/core/Allocator.h 差异（分配器纯虚函数） ==="
diff -u "$V21/include/c10/core/Allocator.h" "$V28/include/c10/core/Allocator.h" | head -60 || true

echo
echo "=== 5. DeviceGuardImplInterface.h 纯虚函数对比 ==="
echo "--- 2.1 的 = 0; ---"
grep -n "= 0;" "$V21/include/c10/core/impl/DeviceGuardImplInterface.h" 2>/dev/null | head -20
echo "--- 2.8 的 = 0; ---"
grep -n "= 0;" "$V28/include/c10/core/impl/DeviceGuardImplInterface.h" | head -20
echo "--- 接口整体行数 ---"
wc -l "$V21/include/c10/core/impl/DeviceGuardImplInterface.h" "$V28/include/c10/core/impl/DeviceGuardImplInterface.h"

echo
echo "=== 6. PrivateUse1HooksInterface.h 是否有（2.1 有 / 2.8 有） ==="
ls -l "$V21/include/ATen/detail/PrivateUse1HooksInterface.h" 2>&1 | tail -1
ls -l "$V28/include/ATen/detail/PrivateUse1HooksInterface.h" 2>&1 | tail -1
echo "--- 2.1 与 2.8 的 PrivateUse1Hooks 纯虚/抛错函数对比 ---"
grep -c "FAIL_PRIVATEUSE1HOOKS_FUNC" "$V21/include/ATen/detail/PrivateUse1HooksInterface.h" 2>/dev/null || echo "2.1: 文件不存在"
grep -c "FAIL_PRIVATEUSE1HOOKS_FUNC" "$V28/include/ATen/detail/PrivateUse1HooksInterface.h" || true

echo
echo "=== 7. Stream / Event 接口签名差异（会影响我们实现交换流与事件） ==="
for h in c10/core/Stream.h c10/core/Event.h c10/core/GeneratorImpl.h; do
    echo "--- $h ---"
    diff -u "$V21/include/$h" "$V28/include/$h" 2>/dev/null | grep -E "^[-+][^-+]" | head -12 || echo "  （无差异或文件不存在）"
done

echo
echo "=== 8. torch.library 的 PrivateUse1 相关能力 ==="
echo "--- 2.1 有没有 Library.fallback / fallthrough_kernel ---"
grep -n "def fallback\|fallthrough_kernel\|def impl(" "$V21/library.py" | head -8
echo "--- 2.8 ---"
grep -n "def fallback\|fallthrough_kernel\|def impl(" "$V28/library.py" | head -8

echo
echo "=== 9. 2.1 里厂商私有的东西（判断哪些是厂商加的、不能当上游 API 抄） ==="
ls "$V21/csrc" 2>/dev/null | head
grep -rn "corex\|iluvatar" "$V21/utils/backend_registration.py" 2>/dev/null | head -5 || echo "  backend_registration.py 里没有 corex 字样"
echo "--- 2.1 里叫 corex 的 python 包 / 模块 ---"
ls -d "$V21/corex" 2>/dev/null || echo "  没有 torch/corex"
find "$V21" -maxdepth 2 -name "*corex*" 2>/dev/null | head -5
echo "PROBE107_DONE"
