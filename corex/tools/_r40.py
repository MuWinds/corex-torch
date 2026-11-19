import io
h='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/TensorTopK.h'
s=io.open(h,encoding='utf-8').read()
i=s.find('#if')
j=s.find('#endif', i)
print('HEADER BLOCK:\n'+s[i:j+7])
if 'template' in s[i:j+7] or 'TensorBase' in s[i:j+7]:
    new='''void launch_gather_topk_kernel(
    const Tensor& self,
    int64_t k, int64_t dim, bool largest,
    const Tensor& values, const Tensor& indices);
void launch_gather_topk_kernel(
    const TensorBase& self,
    int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices);'''
    s=s[:i]+new+s[j+7:]
    io.open(h,'w',encoding='utf-8').write(s); print('header now declares both')