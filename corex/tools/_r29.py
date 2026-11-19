import io, os, re
R='/data/home/2024240384035/build/torch24'
# 1 move compat include out of USE_ROCM block
p=R+'/aten/src/ATen/cuda/CUDABlas.cpp'
s=io.open(p,encoding='utf-8').read()
s=s.replace('#include <rocblas/rocblas.h>\n#include <c10/cuda/CoreXCompat.h>','#include <rocblas/rocblas.h>')
if '#include <c10/cuda/CoreXCompat.h>' not in s.split('#ifdef USE_ROCM')[0]:
    s=s.replace('#include <c10/util/irange.h>','#include <c10/util/irange.h>\n#include <c10/cuda/CoreXCompat.h>',1)
io.open(p,'w',encoding='utf-8').write(s); print('CUDABlas include moved')
# 2 add MTIA_CCP_EVENTS
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): continue
    s=io.open(p,encoding='utf-8').read()
    if 'MTIA_CCP_EVENTS' in s: print('has',d); continue
    s=s.replace('    MTIA_WORKLOADD,','    MTIA_CCP_EVENTS,\n    MTIA_WORKLOADD,',1)
    io.open(p,'w',encoding='utf-8').write(s); print('added',d)