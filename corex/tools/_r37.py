import io
p='/data/home/2024240384035/build/torch24/caffe2/CMakeLists.txt'
s=io.open(p,encoding='utf-8').read()
line='list(APPEND Caffe2_GPU_CU_SRCS ${ATen_CUDA_CU_SRCS})\n'
mark='# CoreX: TensorFactories.cu dropped by glob on shared FS\nlist(APPEND Caffe2_GPU_CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/cuda/TensorFactories.cu)\n'
if 'CoreX: TensorFactories.cu' in s:
    print('already')
else:
    assert line in s
    s=s.replace(line, line+mark,1)
    io.open(p,'w',encoding='utf-8').write(s); print('added')