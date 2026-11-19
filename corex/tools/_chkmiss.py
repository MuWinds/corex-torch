import os, io, subprocess, re
R='/data/home/2024240384035/build/torch24'
nj=io.open(R+'/build/build.ninja',encoding='utf-8',errors='replace').read()
d=R+'/aten/src/ATen/native/cuda'
miss=[]
for f in sorted(os.listdir(d)):
    if not f.endswith('.cu'): continue
    if ('native/cuda/'+f) not in nj: miss.append(f)
print('total', len([f for f in os.listdir(d) if f.endswith('.cu')]), 'missing', len(miss))
print('\n'.join(miss[:20]))