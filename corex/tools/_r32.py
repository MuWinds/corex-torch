import io, os
R='/data/home/2024240384035/build/torch24'
# enums
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): continue
    s=io.open(p,encoding='utf-8').read()
    add=''
    for n in ['PRIVATEUSE1_DRIVER']:
        if n not in s: add+='    %s,\n'%n
    if add:
        s=s.replace('    ENUM_COUNT','%s    ENUM_COUNT'%add,1)
        io.open(p,'w',encoding='utf-8').write(s); print('enum',d)
p=R+'/aten/src/ATen/cuda/CUDABlas.cpp'
s=io.open(p,encoding='utf-8').read()
# 1 guard descriptor classes + lt template
start=s.index('// Following the pattern of CuSparseDescriptor')
endmark='template <typename Dtype>\ninline void bgemm_internal_cublas(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {'
end=s.index(endmark)
block=s[start:end]
stub='''#if defined(__ILUVATAR__)
template <typename Dtype>
inline void bgemm_internal_cublaslt(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "CoreX: cublasLt backend is not supported on this build");
}
#endif

'''
s=s[:start]+'#if !defined(__ILUVATAR__)\n'+block+'#endif\n'+stub+s[end:]
# 2 guard gemm_and_bias body
gs=s.index('void gemm_and_bias(')
# find matching brace
i=s.index('{', gs); depth=0
for j in range(i,len(s)):
    if s[j]=='{': depth+=1
    elif s[j]=='}':
        depth-=1
        if depth==0: break
sig=s[gs:s.index('{',gs)]
s=s[:gs]+'#if !defined(__ILUVATAR__)\n'+s[gs:j+1]+'\n#else\n'+sig+'{ TORCH_CHECK(false, "CoreX: cublasLt not supported"); }\n#endif'+s[j+1:]
io.open(p,'w',encoding='utf-8').write(s); print('cublasLt guarded')