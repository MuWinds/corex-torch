import io
p='/data/home/2024240384035/build/torch24/third_party/kineto-v04/libkineto/include/ActivityType.h'
s=io.open(p,encoding='utf-8').read()
old='    ENUM_COUNT // This is to add buffer'
new='    MTIA_WORKLOADD, // CoreX compat (torch 2.4 expects it)\n    ENUM_COUNT // This is to add buffer'
if 'MTIA_WORKLOADD' in s:
    print('already')
elif old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('enum added')
else:
    print('MISS')