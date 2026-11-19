import io, os
R='/data/home/2024240384035/build/torch24'
p=R+'/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
s=s.replace('add_compile_options(-include ${CMAKE_CURRENT_LIST_DIR}/../../c10/cuda/CoreXCompat.h)\n','')
io.open(p,'w',encoding='utf-8').write(s); print('forced include removed', '-include' not in s)

h=R+'/c10/cuda/CoreXCompat.h'
io.open(h,'w',encoding='utf-8').write('''#pragma once
#if defined(__ILUVATAR__) && defined(__has_include)
#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#endif
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
#endif
#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#ifndef __ILUVATAR_CUBLASLT_OPAQUE
#define __ILUVATAR_CUBLASLT_OPAQUE
typedef cublasLtMatmulDesc_t cublasLtMatmulDescOpaque_t;
#endif
#endif
#endif
''')
print('compat rewritten')

for f in ['aten/src/ATen/cuda/CUDABlas.cpp','aten/src/ATen/cuda/CublasHandlePool.cpp','aten/src/ATen/cuda/CUDABlasLt.cpp']:
    p=os.path.join(R,f)
    if not os.path.exists(p): print('nofile',f); continue
    s=io.open(p,encoding='utf-8').read()
    if 'CoreXCompat.h' in s: print('has',f); continue
    lines=s.split('\n')
    # insert after last leading #include of first block
    idx=0
    for i,l in enumerate(lines[:60]):
        if l.startswith('#include'): idx=i
    lines.insert(idx+1,'#include <c10/cuda/CoreXCompat.h>')
    io.open(p,'w',encoding='utf-8').write('\n'.join(lines))
    print('inserted',f)