import io
h='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.h'
io.open(h,'w',encoding='utf-8').write('''#pragma once
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at {
namespace native {
void launch_gather_topk_kernel(
    const TensorBase& self,
    int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices);
}}
''')
cu='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.cu'
s=io.open(cu,encoding='utf-8').read()
s=s.replace('const Tensor& self','const TensorBase& self').replace('const Tensor& values','const TensorBase& values').replace('const Tensor& indices','const TensorBase& indices')
io.open(cu,'w',encoding='utf-8').write(s); print('reverted to TensorBase')