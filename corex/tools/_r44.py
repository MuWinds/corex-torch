import io
h='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.h'
s=io.open(h,encoding='utf-8').read()
s=s.replace('#include <ATen/core/Tensor.h>','#include <ATen/ATen.h>')
io.open(h,'w',encoding='utf-8').write(s); print(s[:120])