set -eu
# 目的：把厂商 torch 2.1 的「非头文件」适配补到 2.4 上：
#   1) torch/corex.py（corex=True / corex_arch='ivcore10'）
#   2) torch/__init__.py 导入 torch.corex / torch.corex_arch
#   3) torch/cuda/__init__.py 的 is_bf16_supported() 提前返回 True
#   4) CMake 的 CUDA 下限 11.0 → 10.2（CoreX 兼容层只声明 10.2）
# 用法：bash patch_py_cmake_24.sh [源码树，默认 ~/build/torch24]
T=${1:-$HOME/build/torch24}
cd "$T"

printf "corex = True\ncorex_arch = 'ivcore10'\n" > torch/corex.py

python3 - "$T" <<'PY'
import sys

T = sys.argv[1]

# torch/__init__.py：紧跟版本号导入之后挂 corex 标志
p = T + '/torch/__init__.py'
s = open(p).read()
if 'from .corex import corex' not in s:
    anchor = '    from .torch_version import __version__ as __version__\n'
    assert anchor in s, 'torch/__init__.py anchor missing'
    s = s.replace(
        anchor,
        anchor + '\nfrom .corex import corex as corex\nfrom .corex import corex_arch as corex_arch\n',
        1,
    )
    open(p, 'w').write(s)
    print('torch/__init__.py patched')
else:
    print('torch/__init__.py already patched')

# torch/cuda/__init__.py：is_bf16_supported 直接返回 True（厂商 2.1 的唯一 Python 补丁）
p = T + '/torch/cuda/__init__.py'
s = open(p).read()
if 'if torch.corex:' not in s:
    needle = '    if torch.version.hip:\n        return True\n'
    assert needle in s, 'is_bf16_supported anchor missing'
    s = s.replace(needle, needle + '\n    if torch.corex:\n        return True\n', 1)
    open(p, 'w').write(s)
    print('torch/cuda/__init__.py patched')
else:
    print('torch/cuda/__init__.py already patched')
PY

# CMake：允许 CUDA 10.2（CoreX 兼容层）
grep -n 'CUDA_VERSION VERSION_LESS 11.0' cmake/public/cuda.cmake || true
sed -i 's/if(CUDA_VERSION VERSION_LESS 11.0)/if(CUDA_VERSION VERSION_LESS 10.2)/' cmake/public/cuda.cmake

# setup.py：允许跳过 .gitmodules 全量子模块存在性校验（未启用的模块不必下载）
python3 - "$T" <<'PY'
import sys
T = sys.argv[1]
p = T + '/setup.py'
s = open(p).read()
if 'SKIP_SUBMODULE_CHECK' not in s:
    anchor = '    if bool(os.getenv("USE_SYSTEM_LIBS", False)):\n        return\n'
    assert anchor in s, 'setup.py anchor missing'
    s = s.replace(
        anchor,
        '    if bool(os.getenv("SKIP_SUBMODULE_CHECK", False)):\n        return\n\n' + anchor,
        1,
    )
    open(p, 'w').write(s)
    print('setup.py submodule-check skip added')
else:
    print('setup.py already patched')

# cmake：强制用 CUDACXX 指定的 nvcc（torch 只把它映射到 legacy 的 CUDA_NVCC_EXECUTABLE，
#        enable_language(CUDA) 仍会去 toolkit 根目录找真 nvcc）
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
if 'CMAKE_CUDA_COMPILER "$ENV{CUDACXX}"' not in s:
    anchor = 'enable_language(CUDA)'
    assert anchor in s
    s = s.replace(
        anchor,
        'if(DEFINED ENV{CUDACXX})\n'
        '  set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "" FORCE)\n'
        'endif()\n' + anchor,
        1,
    )
    open(p, 'w').write(s)
    print('cuda.cmake compiler force added')
else:
    print('cuda.cmake already patched')
PY

echo "--- 校验 ---"
grep -n 'VERSION_LESS 10.2' cmake/public/cuda.cmake
grep -n 'from .corex import' torch/__init__.py
grep -n 'if torch.corex' torch/cuda/__init__.py
cat torch/corex.py
