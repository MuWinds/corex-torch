import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/TensorIterator.h'
s=io.open(p,encoding='utf-8').read()
old='ScalarType compute_common_dtype(const Device& = kCPU);'
new='ScalarType compute_common_dtype();'
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('ti fixed')
else:
    print('ti ok' if 'ScalarType compute_common_dtype();' in s else 'ti MISS')