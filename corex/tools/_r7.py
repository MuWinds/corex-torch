import io
p='/data/home/2024240384035/build/torch24/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
old='''if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
endif()'''
new='''if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
  set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
  set(CUDA_LIBRARIES "${CUDAToolkit_LIBRARIES}")
  set(CUDA_VERSION "${CUDAToolkit_VERSION}")
endif()'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('bridged')
else:
    print('ok' if 'set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")' in s else 'MISS')