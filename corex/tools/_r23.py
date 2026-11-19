import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu'
s=io.open(p,encoding='utf-8').read()
if '#if !defined(__ILUVATAR__)' not in s[:200]:
    s='#if !defined(__ILUVATAR__)\n'+s+'\n#endif\n'
    io.open(p,'w',encoding='utf-8').write(s); print('nested matmul disabled for CoreX')
else: print('already')