import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu'
s=io.open(p,encoding='utf-8').read()
if 'SPMM_COO_ALG1' in s and 'CoreX: cusparse' not in s:
    a='#define TORCH_ASSERT_ONLY_METHOD_OPERATORS\n'
    b=a+'''#if defined(__ILUVATAR__)
// CoreX: cusparse lacks SPMM_COO_ALG
#ifndef CUSPARSE_SPMM_COO_ALG1
#define CUSPARSE_SPMM_COO_ALG1 1
#endif
#ifndef CUSPARSE_SPMM_COO_ALG2
#define CUSPARSE_SPMM_COO_ALG2 2
#endif
#endif
'''
    io.open(p,'w',encoding='utf-8').write(s.replace(a,b,1)); print('spmm added')
else:
    print('spmm ok' if 'CoreX: cusparse' in s else 'spmm NOUSE')