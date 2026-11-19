import io, re
h='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.h'
io.open(h,'w',encoding='utf-8').write('''#pragma once
#include <cstdint>

namespace at {
class Tensor;
}

namespace at {
namespace native {
void launch_gather_topk_kernel(
    const Tensor& self,
    int64_t k, int64_t dim, bool largest,
    const Tensor& values, const Tensor& indices);
}}
''')
cu='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.cu'
s=io.open(cu,encoding='utf-8').read()
# drop our appended overload
i=s.find('// CoreX: Tensor overload forwarding to TensorBase impl')
if i>0: s=s[:i].rstrip()+'\n'
print('TensorBase occurrences', s.count('TensorBase'))
s=s.replace('const TensorBase&','const Tensor&')
io.open(cu,'w',encoding='utf-8').write(s)
print('unified to Tensor')