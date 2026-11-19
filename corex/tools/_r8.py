import io
p='/data/home/2024240384035/build/torch24/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
if 'CoreX: arch selection handled by launcher' not in s:
    a='''if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")'''
    b='''# CoreX: arch selection handled by launcher (ivcore10), old FindCUDA module is gone.
if(NOT COMMAND cuda_select_nvcc_arch_flags)
  function(cuda_select_nvcc_arch_flags out_var)
    set(${out_var} "" PARENT_SCOPE)
  endfunction()
endif()
if(CUDAToolkit_FOUND)
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")'''
    if a in s:
        io.open(p,'w',encoding='utf-8').write(s.replace(a,b,1)); print('arch fn added')
    else:
        print('ANCHOR MISS')
else:
    print('already')