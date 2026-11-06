set -u
# 探针 104：PrivateUse1 的 C++ 面证据——头文件声明 + 动态库导出符号 + Python 侧 hook 点。
#
# 为什么做这个探测：03-TORCH28-API.md 要回答「上游 CPU 版 wheel 之上能不能挂新设备」。
# 光看到头文件里有 c10::SetAllocator 不算数：wheel 是 -fvisibility 编的，声明存在但
# 符号没导出（或被内联/被裁）时，外部 C++ 扩展一样链不上。所以对每个要用的入口都要
# 拿「头文件行号 + nm 在哪个 .so 里看到的导出符号」两条证据。
#
# 用法（登录节点直接跑）：
#   python tools/remote.py put probes/104_torch28_privateuse1_cpp.sh /data/home/2024240384035/p104.sh
#   python tools/remote.py sh 'bash /data/home/2024240384035/p104.sh > /data/home/2024240384035/p104.out 2>&1'

set -e
VENV=/data/home/2024240384035/venvs/corex28
PY=$VENV/bin/python
SP=$($PY -c 'import torch,os;print(os.path.dirname(os.path.dirname(torch.__file__)))')
TORCH=$SP/torch
INC=$TORCH/include
LIB=$TORCH/lib

echo "=== 0. 动态库清单 ==="
ls -l "$LIB" | awk '{print $5, $9}' | sort -rn | head -12

echo
echo "=== 1. 关键符号在哪个 .so 里导出（nm -DC = 只看动态导出）==="
for so in libc10.so libtorch_cpu.so libtorch.so libtorch_python.so; do
    test -f "$LIB/$so" || { echo "$so: 不存在"; continue; }
    echo "--- $so ---"
    nm -DC --defined-only "$LIB/$so" 2>/dev/null \
        | grep -E "c10::SetAllocator|c10::GetAllocator|register_privateuse1_backend|get_privateuse1_backend|is_privateuse1_backend_registered|c10::impl::DeviceGuardImplRegistry|c10::impl::getDeviceGuardImpl|PrivateUse1Allocator" \
        | head -20
done

echo
echo "=== 1.1 未定义（依赖导入）符号也算证据：外部扩展能不能拿到这些入口 ==="
nm -DC --undefined-only "$LIB/libtorch_cpu.so" 2>/dev/null \
    | grep -E "SetAllocator|publicize|register_privateuse1" | head -10

echo
echo "=== 2. DeviceType.h 的设备名注入声明（95-115 行原文）==="
sed -n '95,115p' "$INC/c10/core/DeviceType.h"

echo
echo "=== 3. Allocator.h：SetAllocator 与 GetAllocator（255-300 行原文）==="
sed -n '255,300p' "$INC/c10/core/Allocator.h"

echo
echo "=== 4. DeviceGuardImplInterface.h：注册宏原文（340-375 行）==="
sed -n '340,375p' "$INC/c10/core/impl/DeviceGuardImplInterface.h"

echo
echo "=== 4.1 DeviceGuardImplInterface 必实现的纯虚函数（=0 的那些）==="
grep -n "= 0;" "$INC/c10/core/impl/DeviceGuardImplInterface.h" | head -20

echo
echo "=== 5. 头文件里所有提到 privateuse1 的地方（看有没有别的官方 hook）==="
grep -rIl "privateuse1\|PrivateUse1" "$INC/c10" "$INC/ATen" "$INC/torch" 2>/dev/null | head -30
echo "--- 逐个文件的命中行（取前 60 条）---"
grep -rIn "privateuse1_backend\|PrivateUse1" "$INC/c10/core/DeviceType.h" "$INC/c10/core/Device.h" \
    "$INC/c10/core/Stream.h" "$INC/c10/core/Event.h" "$INC/c10/core/GeneratorImpl.h" \
    "$INC/c10/core/impl/VirtualGuardImpl.h" "$INC/c10/core/TensorImpl.h" 2>/dev/null | head -40

echo
echo "=== 6. Python 侧 hook 清单 ==="
$PY - <<'PYEOF'
import torch, inspect
print("--- torch/__init__.py::_register_device_module ---")
print(inspect.getsource(torch._register_device_module))
print("--- torch._C 上的 privateuse1 入口 ---")
print([n for n in dir(torch._C) if "privateuse1" in n.lower() or "PrivateUse1" in n])
print("--- torch.utils 上的 privateuse1 入口 ---")
print([n for n in dir(torch.utils) if "privateuse1" in n.lower()])
print("--- torch.library 上的注册入口 ---")
print([n for n in dir(torch.library) if not n.startswith("_")])
PYEOF

echo
echo "=== 7. torch.library 里 PrivateUse1 相关（含 fallback 能不能挂 PrivateUse1）==="
grep -n "PrivateUse1\|def fallback\|def impl\|def _impl\|_dispatch_key_for_device\|register_fake" \
    "$TORCH/library.py" | head -40
echo "--- torch/_library/ 目录（2.8 把实现拆到这里）---"
ls "$TORCH/_library" 2>/dev/null

echo
echo "=== 8. torch.accelerator（2.6+ 的统一加速器 API，PrivateUse1 也能用）==="
ls "$TORCH/accelerator" 2>/dev/null
grep -n "privateuse1\|def current_accelerator\|def is_available\|def device_count" \
    "$TORCH/accelerator/__init__.py" 2>/dev/null | head -25

echo "PROBE104_DONE"
