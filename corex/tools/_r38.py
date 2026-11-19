import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu'
s=io.open(p,encoding='utf-8').read()
stub='''#else
namespace at {
namespace native {
Tensor bmm_nested_cuda(const Tensor& self, const Tensor& mat2) {
  TORCH_CHECK(false, "CoreX: bmm_nested not supported on this build");
}
} // namespace native
} // namespace at
'''
if 'CoreX: bmm_nested not supported' in s:
    print('already')
else:
    assert s.rstrip().endswith('#endif')
    s=s.rstrip()[:-6].rstrip()+'\n\n'+stub+'\n#endif\n'
    io.open(p,'w',encoding='utf-8').write(s); print('stub added')