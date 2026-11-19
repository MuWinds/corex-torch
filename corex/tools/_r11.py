import io, shutil
src='/data/home/2024240384035/build/pytorch-2.4.0/aten/src/ATen/native/cuda/CuFFTPlanCache.h'
dst='/data/home/2024240384035/build/torch24/aten/src/ATen/native/cuda/CuFFTPlanCache.h'
shutil.copyfile(src,dst); print('cufft reverted')