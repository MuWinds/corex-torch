import io
p='/data/home/2024240384035/build/torch24/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
old='''find_package(CUDA)
if(NOT CUDA_FOUND)'''
new='''# CoreX: no real nvcc; old FindCUDA cannot work. Use CUDAToolkit discovery.
find_package(CUDAToolkit)
set(CUDA_FOUND ${CUDAToolkit_FOUND})
if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
endif()
if(NOT CUDA_FOUND)'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('cuda.cmake find fixed')
else:
    print('already ok' if 'find_package(CUDAToolkit)' in s else 'MISS')