import io
p='/data/home/2024240384035/build/torch24/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
old='''find_package(CUDAToolkit)
set(CUDA_FOUND ${CUDAToolkit_FOUND})'''
new='''set(CUDAToolkit_ROOT "/usr/local/corex" CACHE PATH "CoreX toolkit root")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/corex")
find_package(CUDAToolkit)
set(CUDA_FOUND ${CUDAToolkit_FOUND})'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('root fixed')
else:
    print('already' if 'CUDAToolkit_ROOT "/usr/local/corex"' in s else 'MISS')