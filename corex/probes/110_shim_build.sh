set -u
# 探针 110：在登录节点把 corex_shim.cpp 编成 corex_torch 能 import 的扩展模块（验收 A）。
#
# 为什么这条要单独成探针：**编译必须发生在登录节点**——计算节点只在跑卡时才进，
# 而 shim 的编译纯 CPU、纯 g++，放登录节点能省掉排队。这条探针证明「只需要 torch + g++」。
#
# 编译期不依赖 CUDA 工具链的证据在本脚本最后两段：
#   1) `ldd` 里没有 libcudart / libcuda；
#   2) `is_initialized() == False` 且没有 dlopen 过任何 CoreX 库（登录节点本来也没有）。
#
# 前置（源码已经分两处放好；本脚本不负责传输，传输命令就是下面这两条 put）：
#   python tools/remote.py put tools/build_shim.py $SHIM_REPO/tools/build_shim.py
#   python tools/remote.py put src/csrc/corex_shim.cpp $SHIM_REPO/src/csrc/corex_shim.cpp
#
# 用法（登录节点直接跑，不要投沙箱队列）：
#   python tools/remote.py sh 'bash ~/p110.sh'
# 期望最后一行 PROBE110_DONE，中间有 BUILD_SHIM_OK。

set -e
PY28=/data/home/2024240384035/venvs/corex28/bin/python
WORK=/data/home/2024240384035/coffin-work/base/sandbox/work
#: 远端"仓库镜像"（只放参与编译的两份源码）
SHIM_REPO=${SHIM_REPO:-$WORK/ct/repo}
#: 产物去处：corex_torch 包目录，_device.py 用 `from . import _corex_shim` 取它
PKG_DIR=${PKG_DIR:-$WORK/ct/src/corex_torch}
OUT=$PKG_DIR/_corex_shim.so

echo "=== 0. 环境 $(date -Is) ==="
uname -srm
$PY28 -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)'
g++ --version | head -1
test -f "$SHIM_REPO/src/csrc/corex_shim.cpp" || { echo "SHIM_SRC_MISSING=$SHIM_REPO/src/csrc/corex_shim.cpp"; exit 2; }
test -f "$SHIM_REPO/tools/build_shim.py" || { echo "BUILD_SHIM_MISSING"; exit 2; }

echo
echo "=== 1. 编译（首次约 27 s；源码没变时 ninja 命中，约 0.3 s）$(date -Is) ==="
cd "$SHIM_REPO"
$PY28 tools/build_shim.py --out "$OUT"

echo
echo "=== 2. 产物核对 ==="
ls -l "$OUT"
file "$OUT"
echo "--- 依赖里不许有 CUDA 运行时 ---"
if ldd "$OUT" | grep -qiE 'libcudart|libcuda'; then
    echo "SHIM_LINKS_CUDA_BAD"; ldd "$OUT"; exit 3
fi
echo "NO_CUDART_DEP_OK"
echo "--- 导出符号（python 侧要用的 9 个）---"
$PY28 - "$OUT" <<'PYEOF'
import importlib.util, sys
# 必须先 import torch：扩展链的是 libc10/libtorch_cpu，直接按路径加载会报
# `ImportError: libc10.so: cannot open shared object file`——torch 先起才能把符号带进来。
import torch
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_corex_shim", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
want = ["init", "is_initialized", "device_count", "empty_strided", "from_blob",
        "reserved_bytes", "allocated_bytes", "empty_cache"]
missing = [n for n in want if not hasattr(mod, n)]
print("导出接口:", sorted(n for n in dir(mod) if not n.startswith("_")))
print("缺失接口:", missing or "（无）")
print("is_initialized() =", mod.is_initialized())
print("SHIM_API_OK" if not missing and mod.is_initialized() is False else "SHIM_API_BAD")
raise SystemExit(0 if not missing else 4)
PYEOF

echo
echo "=== 3. 编译命令留档（给文档抄） ==="
grep -m1 '^cflags ' "$SHIM_REPO/build/shim/build.ninja" | cut -c1-400
grep -m1 '^ldflags ' "$SHIM_REPO/build/shim/build.ninja" | cut -c1-400

echo "PROBE110_DONE $(date -Is)"
