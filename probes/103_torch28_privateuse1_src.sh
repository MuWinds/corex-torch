set -u
# 探针 103：把 torch 2.8 CPU wheel 里与「新增一个设备后端」有关的源码证据抓出来。
#
# 为什么做这个探测：03-TORCH28-API.md 里每一条「必须提供 X」都要给出源文件 + 行号，
# 光靠记忆写不算证据。本脚本只读不写 torch，输出按主题分段，可直接粘进文档。
#
# 用法（登录节点直接跑）：
#   python tools/remote.py put probes/103_torch28_privateuse1_src.sh ~/p103.sh
#   python tools/remote.py sh 'bash ~/p103.sh'
# 输出重定向到 ~/p103.out 便于整份取回（内容多，别用 tail）。

set -e
VENV=/data/home/2024240384035/venvs/corex28
PY=$VENV/bin/python
SP=$($PY -c 'import torch,os;print(os.path.dirname(os.path.dirname(torch.__file__)))')
TORCH=$SP/torch
INC=$SP/torch/include

echo "=== 0. 版本与路径 ==="
$PY -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)'
echo "site-packages: $SP"
echo "headers:       $INC"
test -d "$INC" || { echo "HEADERS_MISSING"; exit 2; }

echo
echo "=== 1. Python 侧 backend_registration（rename / generate_methods / 契约） ==="
grep -n "def \|_get_device_properties\|_get_device_count\|_get_device_capability\|_is_in_bad_fork\|_init_extension\|_lazy_init\|NotImplementedError\|requires the" \
    "$TORCH/utils/backend_registration.py" | head -80

echo
echo "=== 1.1 rename_privateuse1_backend 的 docstring 原文（契约清单就在里面） ==="
sed -n '/^def rename_privateuse1_backend/,/^def /p' "$TORCH/utils/backend_registration.py" | head -70

echo
echo "=== 1.2 generate_methods_for_privateuse1_backend 的必需函数表 ==="
sed -n '/^def generate_methods_for_privateuse1_backend/,/^def /p' "$TORCH/utils/backend_registration.py" | head -120

echo
echo "=== 2. torch/__init__.py 里的 _register_device_module ==="
grep -n "_register_device_module\|_get_privateuse1_backend_name\|privateuse1_backend_name" "$TORCH/__init__.py" | head -30

echo
echo "=== 3. C++ 侧：SetAllocator / DeviceGuardImplInterface / 注册宏 / 设备名 ==="
grep -rn "SetAllocator" "$INC/c10/core/Allocator.h" | head -10
echo "--- DeviceGuardImplInterface.h 关键声明 ---"
grep -n "struct C10_API DeviceGuardImplInterface\|virtual Device exchangeDevice\|virtual Device getDevice\|virtual Stream exchangeStream\|virtual void destroyEvent\|virtual void record\|C10_DECLARE_REGISTRY\|REGISTER_GUARD_IMPL\|#define C10_REGISTER_GUARD_IMPL" \
    "$INC/c10/core/impl/DeviceGuardImplInterface.h" | head -40
echo "--- 设备名注入点 ---"
grep -rn "get_privateuse1_backend\|register_privateuse1_backend\|is_privateuse1_backend_registered\|TORCH_PRIVATEUSE1" \
    "$INC/c10/core/DeviceType.h" "$INC/c10/core/Device.h" 2>/dev/null | head -30
echo "--- PrivateUse1 相关宏/常量 ---"
grep -rn "PrivateUse1" "$INC/c10/core/DeviceType.h" | head -20

echo
echo "=== 4. TORCH_LIBRARY_IMPL 与 PrivateUse1 dispatch key ==="
grep -n "define TORCH_LIBRARY_IMPL\|DispatchKey::PrivateUse1\|PrivateUse1" "$INC/torch/library.h" | head -20
grep -rn "PrivateUse1" "$INC/c10/core/DispatchKey.h" | head -20

echo
echo "=== 5. 流 / 事件 / 分配器接口（我们将来要实现的 C++ 面） ==="
ls "$INC/c10/core/" | head -60
echo "--- impl 目录 ---"
ls "$INC/c10/core/impl/" | head -40
echo "--- Allocator.h 的 Allocator 虚函数 ---"
grep -n "struct C10_API Allocator\|virtual DataPtr allocate\|virtual DeleterFnPtr raw_deleter\|virtual void copy_data\|virtual bool isDirectMap" "$INC/c10/core/Allocator.h" | head -20

echo
echo "=== 6. torch 2.8 是否还导出 privateuse1 的 Python 侧 helper ==="
$PY - <<'PYEOF'
import torch
names = [
    "_get_privateuse1_backend_name", "_register_device_module",
    "_is_device_type_privateuse1", "_get_custom_class_python_wrapper",
]
for n in names:
    print(f"torch._C.{n}: {hasattr(torch._C, n)}")
print("torch.utils.rename_privateuse1_backend:", hasattr(torch.utils, "rename_privateuse1_backend"))
print("torch.utils.generate_methods_for_privateuse1_backend:",
      hasattr(torch.utils, "generate_methods_for_privateuse1_backend"))
print("default name =", torch._C._get_privateuse1_backend_name())
PYEOF
echo "PROBE103_DONE"
