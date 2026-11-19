import io, os
R='/data/home/2024240384035/build/torch24'
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): continue
    s=io.open(p,encoding='utf-8').read()
    add=''
    for n in ['CUDA_DRIVER','PRIVATEUSE1_RUNTIME']:
        if n not in s: add+='    %s,\n'%n
    if add:
        s=s.replace('    ENUM_COUNT','%s    ENUM_COUNT'%add,1)
        io.open(p,'w',encoding='utf-8').write(s); print('add',d,add.count(','))
h=R+'/c10/cuda/CoreXCompat.h'
s=io.open(h,encoding='utf-8').read()
s=s.replace('typedef int cublasComputeType_t;','typedef cudaDataType_t cublasComputeType_t;')
s=s.replace('#define CUBLAS_COMPUTE_32F 0\n#define CUBLAS_COMPUTE_64F 1\n#define CUBLAS_COMPUTE_32F_FAST_TF32 2\n#define CUBLAS_COMPUTE_16F 3',
 '#define CUBLAS_COMPUTE_32F CUDA_R_32F\n#define CUBLAS_COMPUTE_64F CUDA_R_64F\n#define CUBLAS_COMPUTE_32F_FAST_TF32 CUDA_R_32F\n#define CUBLAS_COMPUTE_16F CUDA_R_16F')
io.open(h,'w',encoding='utf-8').write(s); print('compute typedef fixed')