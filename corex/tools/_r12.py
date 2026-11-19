import io
p='/data/home/2024240384035/build/torch24/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
if 'CoreX: kineto headers are always needed' not in s:
    a='# ---[ CUDA\n'
    b='''# CoreX: kineto headers are always needed (ActivityType.h) even with USE_KINETO=0
include_directories("${CAFFE2_THIRD_PARTY_ROOT}/kineto/libkineto/include")

'''+a
    if a in s:
        io.open(p,'w',encoding='utf-8').write(s.replace(a,b,1)); print('kineto inc added')
    else:
        print('ANCHOR MISS')
else:
    print('already')