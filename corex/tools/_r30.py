import io, os
R='/data/home/2024240384035/build/torch24'
# kineto enums
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): continue
    s=io.open(p,encoding='utf-8').read()
    add=''
    if 'XPU_RUNTIME' not in s: add+='    XPU_RUNTIME,\n'
    if 'MTIA_RUNTIME' not in s: add+='    MTIA_RUNTIME,\n'
    if add:
        s=s.replace('    ENUM_COUNT','%s    ENUM_COUNT'%add,1)
        io.open(p,'w',encoding='utf-8').write(s); print('enum add',d)
    else: print('ok',d)
# compat compute type
h=R+'/c10/cuda/CoreXCompat.h'
s=io.open(h,encoding='utf-8').read()
if 'cublasComputeType_t' not in s:
    s=s.replace('struct cublasLtMatmulDescOpaque_t;','''struct cublasLtMatmulDescOpaque_t;
typedef int cublasComputeType_t;
#ifndef CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F 0
#define CUBLAS_COMPUTE_64F 1
#define CUBLAS_COMPUTE_32F_FAST_TF32 2
#define CUBLAS_COMPUTE_16F 3
#endif''')
    io.open(h,'w',encoding='utf-8').write(s); print('compute type added')
# CUDABlas template types
p=R+'/aten/src/ATen/cuda/CUDABlas.cpp'
s=io.open(p,encoding='utf-8').read()
s=s.replace('cublasLtMatmulDescOpaque_t,\n                                     &cublasLtMatmulDescDestroy','cublasLtMatmulDescStruct,\n                                     &cublasLtMatmulDescDestroy')
s=s.replace('cublasLtMatrixLayoutOpaque_t,\n                                 &cublasLtMatrixLayoutDestroy','cublasLtMatrixLayoutStruct,\n                                 &cublasLtMatrixLayoutDestroy')
s=s.replace('cublasLtMatmulPreferenceOpaque_t,\n                                     &cublasLtMatmulPreferenceDestroy','cublasLtMatmulPreferenceStruct,\n                                     &cublasLtMatmulPreferenceDestroy')
io.open(p,'w',encoding='utf-8').write(s); print('lt structs patched')