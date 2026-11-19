import io
p='/data/home/2024240384035/build/torch24/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
old='''find_package(CUDA)
if(NOT CUDA_FOUND)'''
new='''# CoreX: no real nvcc, so old-style FindCUDA always fails. Use CUDAToolkit instead.
find_package(CUDAToolkit)
set(CUDA_FOUND ${CUDAToolkit_FOUND})
if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
  set(CUDA_VERSION "${CUDAToolkit_VERSION}")
endif()
if(NOT CUDA_FOUND)'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('deps fixed')
else:
    print('deps ok' if 'find_package(CUDAToolkit)' in s else 'deps MISS')