import io, shutil, os
T='/data/home/2024240384035/build/torch24/third_party'
src=T+'/onnx'; dst=T+'/onnx-1.16.1'
if os.path.isdir(src):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
print('copied', os.path.exists(dst+'/CMakeLists.txt'))
p='/data/home/2024240384035/build/torch24/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
s2=s.replace('${CAFFE2_THIRD_PARTY_ROOT}/onnx"','${CAFFE2_THIRD_PARTY_ROOT}/onnx-1.16.1"')
if s2!=s:
    io.open(p,'w',encoding='utf-8').write(s2); print('path patched')
else:
    print('path unchanged')
print([l for l in s.splitlines() if 'onnx' in l and ('add_subdirectory' in l or 'onnx_proto' in l)][:4])