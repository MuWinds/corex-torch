import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu'
s=io.open(p,encoding='utf-8').read()
if not s.startswith('#include <ATen/ATen.h>'):
    s='#include <ATen/ATen.h>\n#include <c10/util/Exception.h>\n'+s
    io.open(p,'w',encoding='utf-8').write(s); print('includes prepended')
else: print('has')