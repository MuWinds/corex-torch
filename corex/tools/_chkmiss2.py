import io, os
R='/data/home/2024240384035/build/torch24'
nj=io.open(R+'/build/build.ninja',encoding='utf-8',errors='replace').read()
base=R+'/aten/src/ATen'
miss=[]; total=0
for root,dirs,files in os.walk(base):
    if not root.endswith('cuda'): continue
    for f in files:
        if not f.endswith('.cu'): continue
        total+=1
        rel=os.path.relpath(os.path.join(root,f), R+'/aten/src/ATen')
        if rel not in nj: miss.append(rel)
print('total',total,'missing',len(miss))
for m in miss: print(m)