import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/cuda/CUDABlas.cpp'
s=io.open(p,encoding='utf-8').read()
old='#if defined(__ILUVATAR__)\ntemplate <typename Dtype>\ninline void bgemm_internal_cublaslt'
new='#if defined(__ILUVATAR__)\n}\ntemplate <typename Dtype>\ninline void bgemm_internal_cublaslt'
assert old in s
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s); print('namespace closed')