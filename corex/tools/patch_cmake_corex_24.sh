set -eu
# 目的：把 torch 2.4 的 CMake 从「CUDA 语言（nvcc）」改成「CXX 源文件 + clang++ -x cuda」。
#   * CoreX 的 nvcc 只是一个打印版本号的 bash 脚本，根本没有编译能力；
#   * CoreX 的真编译器是 /usr/local/corex/bin/clang++（clang 16），支持 -x cuda 与 ivcore10；
#   * 全仓只有 cmake/public/cuda.cmake 一处 enable_language(CUDA)，换掉它即可不触发 NVIDIA 的
#     nvcc 探针；.cu 通过 CMAKE_CXX_SOURCE_FILE_EXTENSIONS 注册成 CXX，编译时由 launcher 改写。
# 用法：bash patch_cmake_corex_24.sh [源码树，默认 ~/build/torch24]
T=${1:-$HOME/build/torch24}
cd "$T"

python3 - "$T" <<'PY'
import sys
import os

T = sys.argv[1]

# 1) .cu 注册成 CXX 扩展名（必须在 project() 之前）
p = T + '/CMakeLists.txt'
s = open(p).read()
marker = '# CoreX: .cu 作为 CXX 源文件'
if marker not in s:
    anchor = 'project(Torch CXX C)\n'
    assert anchor in s, 'project() anchor missing'
    s = s.replace(
        anchor,
        '# CoreX: .cu 作为 CXX 源文件，由 CMAKE_CXX_COMPILER_LAUNCHER 转给 clang++ -x cuda\n'
        'list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)\n\n' + anchor,
        1,
    )
    open(p, 'w').write(s)
    print('CMakeLists: cu extension added')
else:
    print('CMakeLists already patched')

# 1b) project() 会重置 CMAKE_CXX_SOURCE_FILE_EXTENSIONS，必须在它之后再追加一次，
#     否则 .cu 会被当成未知扩展名、整批被 CMake 忽略（只剩生成的 UfuncCUDA_add.cu）。
p = T + '/CMakeLists.txt'
s = open(p).read()
marker2 = 'list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)\n\nproject(Torch CXX C)\n'
if marker2 in s:
    s = s.replace(
        marker2,
        'list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)\n\n'
        'project(Torch CXX C)\n\n'
        '# CoreX: project() 会重置上面的扩展名列表，这里再补一次\n'
        'list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)\n',
        1,
    )
    open(p, 'w').write(s)
    print('CMakeLists: cu extension re-added after project()')
else:
    print('CMakeLists: post-project cu extension already present')

# 2) 替换唯一的 enable_language(CUDA)
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
old = 'enable_language(CUDA)'
new = (
    '# CoreX: 不启用 CUDA 语言（nvcc 是假的，CUDA 语言探针必然失败）。\n'
    '# .cu 已在顶层注册为 CXX 扩展名，实际编译由 corex-cxx-launcher.sh 改写成\n'
    '#   clang++ -x cuda --cuda-gpu-arch=ivcore10\n'
    'set(CMAKE_CUDA_COMPILER "/usr/local/corex/bin/clang++" CACHE FILEPATH "CoreX clang++")\n'
    'set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/corex-cxx-launcher.sh" CACHE STRING "" FORCE)\n'
)
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('cuda.cmake: enable_language(CUDA) replaced')
else:
    print('cuda.cmake: already patched or line missing')

# 2b) 去掉「CUDA 编译器版本 vs CUDAToolkit 版本」一致性检查。
#     没有 CUDA 语言编译器，这个变量天然为空；版本一律以 CUDAToolkit（读 cuda.h）为准。
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
start = s.find('if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)')
if start != -1:
    end = s.find('\nendif()', start)
    assert end != -1
    end += len('\nendif()')
    replacement = (
        '# CoreX: 没有 CUDA 语言编译器，版本统一以 CUDAToolkit（读 cuda.h）为准\n'
        'set(CMAKE_CUDA_COMPILER_VERSION "${CUDAToolkit_VERSION}")\n'
        'set(CUDA_VERSION "${CUDAToolkit_VERSION}" CACHE STRING "" FORCE)\n'
        'set(CUDA_VERSION_STRING "${CUDAToolkit_VERSION}")\n'
    )
    s = s[:start] + replacement + s[end:]
    open(p, 'w').write(s)
    print('cuda.cmake: version check replaced')
else:
    print('cuda.cmake: version check already patched')

# 2c) 去掉 FindCUDA（读假 nvcc）与头文件版本的比对；CoreX 版本一律以头文件为准。
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
start = s.find('if(NOT cuda_version_from_header STREQUAL ${CUDA_VERSION_STRING})')
if start != -1:
    end = s.find('\n    endif()', start)
    assert end != -1
    end += len('\n    endif()')
    s = s[:start] + '# CoreX: 不再比对 FindCUDA（假 nvcc）与头文件版本\n' + s[end:]
    open(p, 'w').write(s)
    print('cuda.cmake: FindCUDA version comparison removed')
else:
    print('cuda.cmake: FindCUDA version comparison already removed')

# 2d) ONNX 的构建开关是 INTERN_DISABLE_ONNX，USE_ONNX=0 管不到它；挂到 USE_ONNX 上。
p = T + '/cmake/Dependencies.cmake'
s = open(p).read()
old = 'if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_DISABLE_ONNX)'
new = 'if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_DISABLE_ONNX AND USE_ONNX)'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('Dependencies.cmake: ONNX gated by USE_ONNX')
else:
    print('Dependencies.cmake: ONNX gate already patched')

# 2e) torch_python 对 onnx_proto 的依赖要加 TARGET 保护（ONNX 已关闭）。
p = T + '/torch/CMakeLists.txt'
s = open(p).read()
old = 'add_dependencies(torch_python onnx_proto)'
new = ('if(TARGET onnx_proto)\n'
       '  add_dependencies(torch_python onnx_proto)\n'
       'endif()')
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('torch/CMakeLists.txt: onnx_proto dependency guarded')
else:
    print('torch/CMakeLists.txt: onnx_proto dependency already guarded')

# 2f) 显式把 CUDA 的 .cu 源列表标成 CXX：不靠扩展名推断（CMake 会在生成阶段
#     把认不出的扩展名当头文件丢掉），确保 torch_cuda 里真的有 kernel 目标文件。
p = T + '/caffe2/CMakeLists.txt'
s = open(p).read()
anchor = '  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)\n  list(APPEND Caffe2_GPU_SRCS ${GENERATED_CXX_TORCH_CUDA})'
if anchor in s and 'CoreX: .cu 一律按 CXX 编译' not in s:
    s = s.replace(
        anchor,
        anchor + '\n'
        '  # CoreX: .cu 一律按 CXX 编译（由 corex-cxx-launcher.sh 转成 clang++ -x cuda）\n'
        '  set_source_files_properties(\n'
        '      ${Caffe2_GPU_CU_SRCS} ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY}\n'
        '      PROPERTIES LANGUAGE CXX)',
        1,
    )
    open(p, 'w').write(s)
    print('caffe2/CMakeLists.txt: CUDA sources forced to CXX')
else:
    print('caffe2/CMakeLists.txt: already patched')

# 2g) clang 的 CUDA 前端会在 host pass 里也语义检查 device 代码，此时 __CUDA_ARCH__ 未定义，
#     c10::guts::apply 会退到 std::apply（host 函数），device 代码里必然编不过。
#     把 __CUDACC__ 一并排除，让 host/device 两遍都走自带的 apply 实现。
p = T + '/c10/util/C++17.h'
s = open(p).read()
old = '#if defined(__cpp_lib_apply) && !defined(__CUDA_ARCH__) && !defined(__HIP__)'
new = '#if defined(__cpp_lib_apply) && !defined(__CUDA_ARCH__) && !defined(__HIP__) && !defined(__CUDACC__)'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('c10/util/C++17.h: CUDACC excluded from std::apply')
else:
    print('c10/util/C++17.h: already patched')

# 2h) CMakeCache 里出现非 UTF-8 字节（本地化路径），setup.py 按 utf-8 读会直接崩。
p = T + '/tools/setup_helpers/cmake.py'
s = open(p).read()
old = 'with open(self._cmake_cache_file) as f:'
new = 'with open(self._cmake_cache_file, encoding="utf-8", errors="replace") as f:'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('cmake.py: tolerant cache decode')
else:
    print('cmake.py: already patched')

# 2i) env.py 里还有一处直接 open CMakeCache（决定 BUILD_TYPE），同样要容错。
p = T + '/tools/setup_helpers/env.py'
s = open(p).read()
old = 'with open(cmake_cache_txt) as f:'
new = 'with open(cmake_cache_txt, encoding="utf-8", errors="replace") as f:'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('env.py: tolerant cache decode')
else:
    print('env.py: already patched')

# 2j) CoreX 的 NVML 头没有 nvmlDeviceGetNvLinkRemoteDeviceType（2.1 时代还没这个符号）。
p = T + '/c10/cuda/driver_api.h'
s = open(p).read()
start = s.find('#define C10_NVML_DRIVER_API(')
if start != -1 and '__ILUVATAR__' not in s[start:start + 400]:
    end = s.find(')', s.find('_(nvmlDeviceGetComputeRunningProcesses)', start))
    end = s.find('\n', end)
    old_block = s[start:end]
    new_block = (
        '#ifdef __ILUVATAR__\n'
        '#define C10_NVML_DRIVER_API(_)           \\\n'
        '  _(nvmlInit_v2)                         \\\n'
        '  _(nvmlDeviceGetHandleByPciBusId_v2)    \\\n'
        '  _(nvmlDeviceGetNvLinkRemotePciInfo_v2) \\\n'
        '  _(nvmlDeviceGetComputeRunningProcesses)\n'
        '#else\n'
        + old_block +
        '\n#endif'
    )
    s = s[:start] + new_block + s[end:]
    open(p, 'w').write(s)
    print('c10/cuda/driver_api.h: NVLinkRemoteDeviceType guarded')
else:
    print('c10/cuda/driver_api.h: already patched')

# 2k) 厂商的头文件补丁全部挂在 __ILUVATAR__ 上，但 clang/g++ 都不预定义它，
#     必须在编译选项里全局给出，否则 host 侧仍然走 NVIDIA 分支。
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
extra = ('add_definitions(-D__ILUVATAR_WORKAROUND__)\n'
         'add_definitions(-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ '
         '-D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__)')
marker = 'add_definitions(-D__ILUVATAR__)'
if marker not in s:
    anchor = 'set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/corex-cxx-launcher.sh" CACHE STRING "" FORCE)'
    s = s.replace(anchor, anchor + '\n\n# CoreX: 打开厂商补丁用的宏\n' + marker + '\n' + extra, 1)
    open(p, 'w').write(s)
    print('cuda.cmake: __ILUVATAR__ defined globally')
elif '__ILUVATAR_WORKAROUND__' not in s:
    s = s.replace(marker, marker + '\n' + extra, 1)
    open(p, 'w').write(s)
    print('cuda.cmake: workaround/CUDA_NO_HALF macros added')
else:
    print('cuda.cmake: macros already defined')

# 2l) CoreX 的 NVML 只有 nvmlProcessInfo_t，没有 2.4 用的 v1 类型。
p = T + '/c10/cuda/CUDACachingAllocator.cpp'
s = open(p).read()
marker = 'using nvmlProcessInfo_v1_t = nvmlProcessInfo_t;'
if marker not in s:
    anchor = 'namespace c10 {'
    idx = s.find(anchor)
    assert idx != -1
    s = s[:idx] + '#ifdef __ILUVATAR__\nusing nvmlProcessInfo_v1_t = nvmlProcessInfo_t;\n#endif\n\n' + s[idx:]
    open(p, 'w').write(s)
    print('CUDACachingAllocator.cpp: nvmlProcessInfo_v1_t aliased')
else:
    print('CUDACachingAllocator.cpp: already patched')

# 2m) CoreX 的 cuda_wrappers/algorithm 提供 min 模板，0.5 与 double 混用会推导冲突。
p = T + '/c10/util/order_preserving_flat_hash_map.h'
s = open(p).read()
old = 'std::min(0.5, static_cast<double>(_max_load_factor))'
new = 'std::min<double>(0.5, static_cast<double>(_max_load_factor))'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('order_preserving_flat_hash_map.h: std::min<double>')
else:
    print('order_preserving_flat_hash_map.h: already patched')

# 2n) 同一处 std::min 在 flat_hash_map.h 里还有一份。
p = T + '/c10/util/flat_hash_map.h'
s = open(p).read()
old = 'std::min(0.5, static_cast<double>(_max_load_factor))'
new = 'std::min<double>(0.5, static_cast<double>(_max_load_factor))'
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('flat_hash_map.h: std::min<double>')
else:
    print('flat_hash_map.h: already patched')

# 2o) CoreX 的 isnan 是 __device__-only，_isnan 是 host+device，host pass 会报错。
p = T + '/aten/src/ATen/NumericUtils.h'
s = open(p).read()
old = '''#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);'''
new = '''#if defined(__ILUVATAR__)
  return val != val;
#elif defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('NumericUtils.h: isnan replaced for CoreX')
else:
    print('NumericUtils.h: already patched')

# 2p) __ILUVATAR_WORKAROUND__ 下 Atomic.cuh 的 Mul/Max/Min 变成 3 参数（多一个 mutex），
#     但 2.4 的调用点还是 2 参数，补一层转发模板。
p = T + '/aten/src/ATen/cuda/Atomic.cuh'
s = open(p).read()
marker = 'COREX_ATOMIC_2ARG_FORWARD'
if marker not in s:
    s += '''
// CoreX: __ILUVATAR_WORKAROUND__ 版本多一个 mutex 参数，这里给 2 参数调用补转发
#ifdef __ILUVATAR_WORKAROUND__
#define COREX_ATOMIC_2ARG_FORWARD(NAME)                                                         \\
template <typename T>                                                                          \\
inline __device__ auto gpuAtomic##NAME(T* a, T v)                                              \\
    -> decltype(gpuAtomic##NAME(a, v, static_cast<int32_t*>(nullptr))) {                       \\
  return gpuAtomic##NAME(a, v, static_cast<int32_t*>(nullptr));                                \\
}
COREX_ATOMIC_2ARG_FORWARD(Mul)
COREX_ATOMIC_2ARG_FORWARD(Max)
COREX_ATOMIC_2ARG_FORWARD(Min)
#undef COREX_ATOMIC_2ARG_FORWARD
#endif
'''
    open(p, 'w').write(s)
    print('Atomic.cuh: 2-arg forwards added')
else:
    print('Atomic.cuh: already patched')

# 2q) cutlass 的 printf 在 -Werror=format 下会直接报错，降为警告。
p = T + '/cmake/public/cuda.cmake'
s = open(p).read()
if '-Wno-error=format' not in s:
    anchor = 'add_definitions(-D__ILUVATAR__)'
    s = s.replace(anchor, anchor + '\nadd_compile_options(-Wno-error=format)', 1)
    open(p, 'w').write(s)
    print('cuda.cmake: -Wno-error=format')
else:
    print('cuda.cmake: format warning already relaxed')

# 2r) _isinf 同一问题：CoreX 的 isinf 是 __device__-only。
p = T + '/aten/src/ATen/NumericUtils.h'
s = open(p).read()
old = '''#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);'''
new = '''#if defined(__ILUVATAR__)
  return val > std::numeric_limits<T>::max() || val < -std::numeric_limits<T>::max();
#elif defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('NumericUtils.h: isinf replaced for CoreX')
else:
    print('NumericUtils.h: isinf already patched')

# 2s) CoreX 的 cudaDeviceProp 没有 maxBlocksPerMultiProcessor。
p = T + '/aten/src/ATen/native/cuda/TensorTopK.cu'
s = open(p).read()
old = '''#if !defined(USE_ROCM)
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;'''
new = '''#if !defined(USE_ROCM) && !defined(__ILUVATAR__)
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('TensorTopK.cu: maxBlocksPerMultiProcessor guarded')
else:
    print('TensorTopK.cu: already guarded')

# 2t) cutlass 的 cast_smem_ptr_to_uint 用了 CoreX clang 没有的 __cvta_generic_to_shared。
p = T + '/third_party/cutlass/include/cute/arch/util.hpp'
s = open(p).read()
old = '''#if CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
  //'''
new = '''#if defined(__ILUVATAR__)
  // CoreX: clang 不提供 __cvta_generic_to_shared，shared 指针本身就是线性地址。
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
#elif CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
  //'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('cutlass util.hpp: cvta fallback for CoreX')
else:
    print('cutlass util.hpp: already patched')

# 2u) cutlass 的 first<> 用可变参数 device 函数，clang 不支持。
p = T + '/third_party/cutlass/include/cutlass/functional.h'
s = open(p).read()
old = '''  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {
    return first;
  }'''
new = '''#if defined(__ILUVATAR__)
  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &) const {
    return first;
  }
  CUTLASS_HOST_DEVICE
  T operator()(T const & first) const {
    return first;
  }
#else
  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {
    return first;
  }
#endif'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('cutlass functional.h: variadic replaced for CoreX')
else:
    print('cutlass functional.h: already patched')

# 2v) EmbeddingBag.cu 里还有一份 bfe.u32 的 inline asm，CoreX 汇编器不认。
p = T + '/aten/src/ATen/native/quantized/cuda/EmbeddingBag.cu'
s = open(p).read()
old = '''__device__ __forceinline__ float bfe(uint32_t val, uint32_t pos, uint32_t len) {
#ifdef USE_ROCM'''
new = '''__device__ __forceinline__ float bfe(uint32_t val, uint32_t pos, uint32_t len) {
#if defined(USE_ROCM) || defined(__ILUVATAR__)'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('EmbeddingBag.cu: bfe software path for CoreX')
else:
    print('EmbeddingBag.cu: already patched')

# 2w) CoreX cuSPARSE 没有 COO_ALG1/2 这两个枚举名。
p = T + '/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu'
s = open(p).read()
old = '  cusparseSpMMAlg_t mm_alg = deterministic ? CUSPARSE_SPMM_COO_ALG2 : CUSPARSE_SPMM_COO_ALG1;'
new = '''#ifdef __ILUVATAR__
  cusparseSpMMAlg_t mm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
#else
  cusparseSpMMAlg_t mm_alg = deterministic ? CUSPARSE_SPMM_COO_ALG2 : CUSPARSE_SPMM_COO_ALG1;
#endif'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('SparseCUDATensorMath.cu: cusparse enum fallback for CoreX')
else:
    print('SparseCUDATensorMath.cu: already patched')

# 2x) CoreX 的 thrust/mr/new.h 在 C++17 下调用 sized aligned delete，
#     clang 的 device pass 找不到该重载；放一个 shim 头把它退回旧分支。
shim_dir = T + '/cmake/public/corex-shim/thrust/mr'
os.makedirs(shim_dir, exist_ok=True)
src_thrust = '/usr/local/corex/include/thrust/mr/new.h'
if os.path.exists(src_thrust):
    txt = open(src_thrust).read()
    txt2 = txt.replace('#if __cplusplus >= 201703L', '#if 0  // CoreX: clang device pass 无 sized aligned delete')
    open(shim_dir + '/new.h', 'w').write(txt2)
    print('thrust shim: new.h written')
else:
    print('thrust shim: corex header not visible, skipped')

# 3) launcher：只在命令行出现 .cu 时切成 clang++ -x cuda
launcher = T + '/cmake/public/corex-cxx-launcher.sh'
open(launcher, 'w').write('''set -u
compiler=$1
shift

is_cu=0
for a in "$@"; do
    case "$a" in
        *.cu) is_cu=1; break ;;
    esac
done

if [ "$is_cu" = "1" ]; then
    CX=/usr/local/corex
    CUDA_PATH=/usr/local/corex-3.2.1
    [ -d "$CUDA_PATH" ] || CUDA_PATH=$CX
    SHIM="$(dirname "$0")/corex-shim"
    exec "$CX/bin/clang++" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path="$CUDA_PATH" -I "$SHIM" "$@"
fi

exec "$compiler" "$@"
''')
print('launcher written')
PY

chmod +x cmake/public/corex-cxx-launcher.sh

echo "--- 校验 ---"
grep -n 'CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu' CMakeLists.txt
grep -n 'corex-cxx-launcher' cmake/public/cuda.cmake
ls -l cmake/public/corex-cxx-launcher.sh
