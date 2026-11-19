import io, os
R='/data/home/2024240384035/build/torch24'
# compat header simple
io.open(R+'/c10/cuda/CoreXCompat.h','w',encoding='utf-8').write('''#pragma once
#if defined(__ILUVATAR__)
#include <cublas_v2.h>
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#endif
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
struct cublasLtMatmulDescOpaque_t;
#endif
''')
print('compat simple')
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): print('skip',d); continue
    s=io.open(p,encoding='utf-8').read()
    if 'MTIA_WORKLOADD' in s and 'CUDA_SYNC' in s: print('ok',d); continue
    i=s.find('enum class ActivityType')
    j=s.find('};', i)
    add=''
    if 'MTIA_WORKLOADD' not in s: add+='    MTIA_WORKLOADD,\n'
    if 'CUDA_SYNC' not in s: add+='    CUDA_SYNC,\n'
    if j>0:
        s=s[:j]+add+s[j:]
        io.open(p,'w',encoding='utf-8').write(s); print('patched',d)
    else: print('no enum',d)