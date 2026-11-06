set -u
# 探针 105：PrivateUse1 的「第二层」入口——ATen 的 PrivateUse1Hooks / 生成器 / device lazy init，
# 以及 C++ 侧真正需要 link 的符号是否导出（决定 C++ shim 能不能用官方 API 写成）。
#
# 为什么做这个探测：104 只确认了分配器与 guard 的入口。真正接一个设备还会撞上
# rand/generator（torch.randn 走 getDefaultGenerator）、以及 torch 的 lazy init
# （device_lazy_init 在每次 op 前调用，PrivateUse1 需要注册 hooks 才能走过）。
# 这些是不是「最少必需」，必须看源码才知道，不能凭印象写进文档。
#
# 用法（登录节点直接跑）：
#   python tools/remote.py put probes/105_torch28_pu1_hooks.sh /data/home/2024240384035/p105.sh
#   python tools/remote.py sh 'bash /data/home/2024240384035/p105.sh > /data/home/2024240384035/p105.out 2>&1'

set -e
VENV=/data/home/2024240384035/venvs/corex28
PY=$VENV/bin/python
SP=$($PY -c 'import torch,os;print(os.path.dirname(os.path.dirname(torch.__file__)))')
TORCH=$SP/torch
INC=$TORCH/include
LIB=$TORCH/lib

echo "=== 1. C++ shim 要 link 的符号（nm -DC 导出） ==="
for pat in "DeviceGuardImplRegistrar" "Dispatcher" "torch::Library" "PrivateUse1Hooks" "registerImpl" "DeviceTypeName"; do
    echo "--- pattern: $pat ---"
    nm -DC --defined-only "$LIB/libc10.so" "$LIB/libtorch_cpu.so" 2>/dev/null \
        | grep -F "$pat" | head -6
done

echo
echo "=== 2. ATen/detail/PrivateUse1HooksInterface.h 全文 ==="
cat "$INC/ATen/detail/PrivateUse1HooksInterface.h"

echo
echo "=== 3. ATen/core/GeneratorForPrivateuseone.h 关键部分 ==="
grep -n "class \|virtual \|TORCH_DECLARE_REGISTRY\|registerGenerator\|getPrivateUse1DefaultGenerator\|createGeneratorPrivateUse1" \
    "$INC/ATen/core/GeneratorForPrivateuseone.h" | head -40

echo
echo "=== 4. torch/csrc/utils/device_lazy_init.h（PrivateUse1 的 lazy init 入口）==="
cat "$INC/torch/csrc/utils/device_lazy_init.h"

echo
echo "=== 5. torch.library.Library.fallback 实现（能不能给 PrivateUse1 挂 fallback）==="
sed -n '391,470p' "$TORCH/library.py"

echo
echo "=== 6. _get_custom_mod_func 在 python 里被谁用（= device module 的必需函数清单）==="
grep -rn "_get_custom_mod_func" "$TORCH" --include=*.py | grep -v "backend_registration.py" | head -20
echo "--- backend_registration.py 内部（390-430 行）---"
sed -n '390,430p' "$TORCH/utils/backend_registration.py"

echo
echo "=== 7. PrivateUse1 的 lazy init 在 python 侧怎么触发 ==="
grep -rn "device_lazy_init\|_lazy_init" "$TORCH/__init__.py" "$TORCH/_tensor.py" 2>/dev/null | head -15
grep -rn "PrivateUse1\|privateuseone\|privateuse1" "$TORCH/__init__.py" | head -20

echo
echo "=== 8. 实测：dispatch key 与 device module 契约 ==="
$PY - <<'PYEOF'
import torch
print("default pu1 name:", torch._C._get_privateuse1_backend_name())
print("_dispatch_key_for_device('privateuseone'):", torch._C._dispatch_key_for_device("privateuseone"))
print("torch.device('privateuseone'):", torch.device("privateuseone"))
print("has torch.accelerator:", hasattr(torch, "accelerator"))
print("accelerator.current_accelerator():", torch.accelerator.current_accelerator())
try:
    print("accelerator.is_available():", torch.accelerator.is_available())
except Exception as e:
    print("accelerator.is_available() ->", type(e).__name__, e)
PYEOF
echo "PROBE105_DONE"
