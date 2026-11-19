import io
block='''#if defined(__ILUVATAR__)
// CoreX: cusparse 10.2 lacks SPMM_CSR alphabets
#ifndef CUSPARSE_SPMM_CSR_ALG1
#define CUSPARSE_SPMM_CSR_ALG1 1
#endif
#ifndef CUSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_SPMM_CSR_ALG2 2
#endif
#endif
'''
for p in ['/data/home/2024240384035/build/torch24/aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp',
          '/data/home/2024240384035/build/torch24/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu']:
    s=io.open(p,encoding='utf-8').read()
    if 'CUSPARSE_SPMM_CSR_ALG1' in s and 'SPMM_CSR_ALG1\n#define' not in s and 'CoreX: cusparse 10.2 lacks SPMM_CSR' not in s:
        a='#define TORCH_ASSERT_ONLY_METHOD_OPERATORS\n'
        if a in s:
            s=s.replace(a,a+block,1); io.open(p,'w',encoding='utf-8').write(s); print('added',p.split('/')[-1])
        else: print('no anchor',p)
    else: print('skip',p.split('/')[-1], 'CUSPARSE_SPMM_CSR_ALG1' in s)