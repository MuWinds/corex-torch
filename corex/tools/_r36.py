import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/cuda/CUDABlas.cpp'
s=io.open(p,encoding='utf-8').read()
gs=s.index('void int8_gemm(')
i=s.index('{',gs); depth=0
for j in range(i,len(s)):
    if s[j]=='{': depth+=1
    elif s[j]=='}':
        depth-=1
        if depth==0: break
sig=s[gs:s.index('{',gs)]
s=s[:gs]+'#if !defined(__ILUVATAR__)\n'+s[gs:j+1]+'\n#else\n'+sig+'{ TORCH_CHECK(false, "CoreX: int8 gemm via cublasLt not supported"); }\n#endif'+s[j+1:]
io.open(p,'w',encoding='utf-8').write(s); print('int8_gemm guarded')