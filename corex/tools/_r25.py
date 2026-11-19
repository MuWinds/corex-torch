import io, os
R='/data/home/2024240384035/build/torch24'
# compat header: gcc doesn't treat __has_include as macro
h=R+'/c10/cuda/CoreXCompat.h'
s=io.open(h,encoding='utf-8').read()
s=s.replace('#if defined(__ILUVATAR__) && defined(__has_include)','#if defined(__ILUVATAR__)')
io.open(h,'w',encoding='utf-8').write(s)
print('compat guard fixed')
# add MTIA to every ActivityType.h copy
for d in ['kineto','kineto-src','kineto-v04']:
    p=os.path.join(R,'third_party',d,'libkineto/include/ActivityType.h')
    if not os.path.exists(p): print('skip',d); continue
    t=io.open(p,encoding='utf-8').read()
    if 'MTIA_WORKLOADD' in t: print('has',d); continue
    old='    ENUM_COUNT // This is to add buffer'
    if old in t:
        io.open(p,'w',encoding='utf-8').write(t.replace(old,'    MTIA_WORKLOADD,\n'+old,1)); print('added',d)
    else: print('miss',d)