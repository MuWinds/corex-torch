import io
h='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.h'
io.open(h,'w',encoding='utf-8').write('''#pragma once
#include <cstdint>

namespace at {
class Tensor;
class TensorBase;
}

namespace at {
namespace native {
void launch_gather_topk_kernel(
    const TensorBase& self,
    int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices);
void launch_gather_topk_kernel(
    const Tensor& self,
    int64_t k, int64_t dim, bool largest,
    const Tensor& values, const Tensor& indices);
}}
''')
cu='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.cu'
s=io.open(cu,encoding='utf-8').read()
if 'CoreX: Tensor overload' not in s:
    s=s+'''

// CoreX: Tensor overload forwarding to TensorBase impl
namespace at { namespace native {
void launch_gather_topk_kernel(
    const Tensor& self, int64_t k, int64_t dim, bool largest,
    const Tensor& values, const Tensor& indices) {
  launch_gather_topk_kernel(static_cast<const TensorBase&>(self), k, dim, largest,
                            static_cast<const TensorBase&>(values),
                            static_cast<const TensorBase&>(indices));
}
}}
'''
    io.open(cu,'w',encoding='utf-8').write(s); print('overload appended')