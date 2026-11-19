import io, os
R='/data/home/2024240384035/build/torch24/third_party'
old='''    MTIA_WORKLOADD, // CoreX compat (torch 2.4 expects it)
    ENUM_COUNT // This is to add buffer and not used for any profiling logic. Add your new type before it.
    CUDA_SYNC,
};'''
new='''    MTIA_WORKLOADD, // CoreX compat (torch 2.4 expects it)
    CUDA_SYNC,
    ENUM_COUNT // This is to add buffer and not used for any profiling logic. Add your new type before it.
};'''
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): print('skip',d); continue
    s=io.open(p,encoding='utf-8').read()
    if old in s:
        io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('fixed',d)
    elif 'CUDA_SYNC,\n    ENUM_COUNT' in s: print('ok',d)
    else: print('check',d)