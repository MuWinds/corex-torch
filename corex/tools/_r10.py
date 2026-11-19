import io
p='/data/home/2024240384035/build/torch24/torch/csrc/profiler/kineto_shim.h'
s=io.open(p,encoding='utf-8').read()
old='#include <ActivityType.h>\n'
new='#ifdef USE_KINETO\n#include <ActivityType.h>\n#endif\n'
if '#ifdef USE_KINETO\n#include <ActivityType.h>' in s:
    print('already')
elif old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new,1)); print('kineto fixed')
else:
    print('MISS')