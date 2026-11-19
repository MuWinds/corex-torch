import io
h='/data/home/2024240384035/build/torch24/c10/cuda/CoreXCompat.h'
s=io.open(h,encoding='utf-8').read()
if 'CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION' not in s:
    s=s.replace('#endif\n','#ifndef CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\n#define CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION 0\n#endif\n#endif\n',1)
    io.open(h,'w',encoding='utf-8').write(s); print('added')
else: print('has')