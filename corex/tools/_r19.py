import io, os, shutil
R='/data/home/2024240384035/build/torch24'

# 1 cute variadic fallback
p=R+'/third_party/cutlass/include/cute/util/type_traits.hpp'
s=io.open(p,encoding='utf-8').read()
old='''template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto
is_valid_impl(...) { return CUTE_STL_NAMESPACE::false_type{}; }'''
new='''template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto
#if defined(__ILUVATAR__)
is_valid_impl(long) { return CUTE_STL_NAMESPACE::false_type{}; }
#else
is_valid_impl(...) { return CUTE_STL_NAMESPACE::false_type{}; }
#endif'''
print('cute', 'ok' if old in s else 'MISS')
if old in s: io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1))

# 2 cusparse fallback casts
for p in [R+'/aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp', R+'/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu']:
    s=io.open(p,encoding='utf-8').read()
    s=s.replace('#define CUSPARSE_SPMM_COO_ALG1 1','#define CUSPARSE_SPMM_COO_ALG1 ((cusparseSpMMAlg_t)1)')
    s=s.replace('#define CUSPARSE_SPMM_COO_ALG2 2','#define CUSPARSE_SPMM_COO_ALG2 ((cusparseSpMMAlg_t)2)')
    s=s.replace('#define CUSPARSE_SPMM_CSR_ALG1 1','#define CUSPARSE_SPMM_CSR_ALG1 ((cusparseSpMMAlg_t)1)')
    s=s.replace('#define CUSPARSE_SPMM_CSR_ALG2 2','#define CUSPARSE_SPMM_CSR_ALG2 ((cusparseSpMMAlg_t)2)')
    io.open(p,'w',encoding='utf-8').write(s)
print('cusparse casts ok')

# 3 compat header
h=R+'/c10/cuda/CoreXCompat.h'
io.open(h,'w',encoding='utf-8').write('''#pragma once
#if defined(__ILUVATAR__)
#include <cublas_v2.h>
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#endif
#if defined(__has_include)
#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#ifndef __ILUVATAR_CUBLASLT_OPAQUE
#define __ILUVATAR_CUBLASLT_OPAQUE
typedef cublasLtMatmulDesc_t cublasLtMatmulDescOpaque_t;
#endif
#endif
#endif
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
#endif
''')
p=R+'/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
if 'CoreXCompat.h' not in s:
    s=s.replace('add_compile_options(-fsized-deallocation)','add_compile_options(-fsized-deallocation)\nadd_compile_options(-include ${CMAKE_CURRENT_LIST_DIR}/../../c10/cuda/CoreXCompat.h)',1)
    io.open(p,'w',encoding='utf-8').write(s); print('compat forced include added')
else: print('compat already')

# 4 kineto fresh dir
T=R+'/third_party'
if os.path.isdir(T+'/kineto'):
    shutil.rmtree(T+'/kineto-src', ignore_errors=True); shutil.copytree(T+'/kineto', T+'/kineto-src')
p=R+'/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
s2=s.replace('kineto/libkineto/include','kineto-src/libkineto/include')
if s2!=s: io.open(p,'w',encoding='utf-8').write(s2); print('kineto path fresh')
else: print('kineto path already')
print('kineto hdr', os.path.exists(T+'/kineto-src/libkineto/include/ActivityType.h'))