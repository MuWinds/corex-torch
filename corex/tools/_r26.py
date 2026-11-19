import shutil, os
R='/data/home/2024240384035/build/torch24/third_party'
src=R+'/kineto-v04/libkineto/include/ActivityType.h'
for d in ['kineto','kineto-src']:
    dst=os.path.join(R,d,'libkineto/include/ActivityType.h')
    if os.path.isdir(os.path.dirname(dst)):
        shutil.copyfile(src,dst); print('copied into',d)
    else: print('nodir',d)