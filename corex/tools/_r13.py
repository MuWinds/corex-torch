import io
p='/data/home/2024240384035/build/torch24/third_party/onnx/CMakeLists.txt'
s=io.open(p,encoding='utf-8',errors='replace').read()
io.open(p,'w',encoding='utf-8').write(s)
print('rewrote', len(s))
import os
print('exists', os.path.exists(p), os.path.getsize(p))