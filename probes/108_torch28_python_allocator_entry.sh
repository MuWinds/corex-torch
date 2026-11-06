set -u
# 探针 108：Python 侧到底有没有「设置 PrivateUse1 分配器」的入口？
#
# 为什么做这个探测：这是「纯 Python 收工」与「先写 C++ 扩展」的分岔口。判据不能是印象，
# 得把三处都翻一遍：① torch._C 上所有 privateuse / alloc 相关名字；② 类型存根 _C/*.pyi；
# ③ Python 源码里的 privateuse1 命中点，逐个看是不是分配器入口。
# 另外附带确认 rename("corex") 之后 torch.cuda.is_available() 仍为 False（我们的设备
# 不能把 cuda 弄成可用，否则所有 "is cuda available" 的分支都会走错）。
#
# 用法（登录节点直接跑）：
#   python tools/remote.py put probes/108_torch28_python_allocator_entry.sh /data/home/2024240384035/p108.sh
#   python tools/remote.py sh 'bash /data/home/2024240384035/p108.sh > /data/home/2024240384035/p108.out 2>&1'

set -e
VENV=/data/home/2024240384035/venvs/corex28
PY=$VENV/bin/python
TORCH=$VENV/lib/python3.10/site-packages/torch

echo "=== 1. Python 源码里 privateuse1 的命中点（去掉纯文档/docstring 的部分）==="
grep -rn "privateuse1" "$TORCH" --include=*.py 2>/dev/null | grep -viE "^.*:[0-9]+: *(#|\"|')" | head -40

echo
echo "=== 2. Python 源码里出现 allocator 的私有 API（全 torch 包）==="
grep -rn "def .*allocator\|set_allocator\|SetAllocator\|_get_allocator" "$TORCH" --include=*.py 2>/dev/null | head -25

echo
echo "=== 3. 类型存根 _C/*.pyi 里 privateuse / allocator 相关声明 ==="
ls "$TORCH/_C/"*.pyi 2>/dev/null
grep -rn "privateuse\|allocator" "$TORCH/_C/"*.pyi 2>/dev/null | head -30

echo
echo "=== 4. 运行时 torch._C 上的名字（这是最硬的证据：没有绑定就是没有）==="
"$PY" - <<'PYEOF'
import torch
pu = sorted(n for n in dir(torch._C) if "privateuse" in n.lower())
print("torch._C 里 privateuse* :", pu)
al = sorted(n for n in dir(torch._C) if "alloc" in n.lower())
print("torch._C 里 *alloc*     :", al)
u = sorted(n for n in dir(torch.utils) if "privateuse" in n.lower())
print("torch.utils 里 privateuse* :", u)
# Library / 其他可能藏着入口的对象
print("torch.library 里 alloc 相关:", [n for n in dir(torch.library) if "alloc" in n.lower()])
PYEOF

echo
echo "=== 5. rename 之后 cuda 相关状态是否被污染 ==="
"$PY" - <<'PYEOF'
import torch
print("rename 之前: cuda.is_available =", torch.cuda.is_available())
print("rename 之前: cuda.device_count =", torch.cuda.device_count())
torch.utils.rename_privateuse1_backend("corex")
print("rename 之后: privateuse1 name  =", torch._C._get_privateuse1_backend_name())
print("rename 之后: torch.device('corex') =", torch.device("corex"))
print("rename 之后: cuda.is_available =", torch.cuda.is_available())
print("rename 之后: cuda.device_count =", torch.cuda.device_count())
print("rename 之后: accelerator.current_accelerator =", torch.accelerator.current_accelerator())
# 下面两条会失败，用 try 兜住：探针要跑到最后一行，不能因为预期内的失败中断。
for label, fn in (
    ("accelerator.is_available", torch.accelerator.is_available),
    ("torch.empty(4, device='corex')", lambda: torch.empty(4, device="corex")),
):
    try:
        print(f"rename 之后: {label} -> {fn()}")
    except BaseException as exc:  # noqa: BLE001 - 预期内失败
        print(f"rename 之后: {label} -> {type(exc).__name__}: {str(exc).splitlines()[0]}")
PYEOF
echo "PROBE108_DONE"
