import io, shutil, os
T='/data/home/2024240384035/build/torch24/third_party'
shutil.rmtree(T+'/foxi-src', ignore_errors=True); shutil.copytree(T+'/foxi', T+'/foxi-src')
p='/data/home/2024240384035/build/torch24/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
a='third_party/foxi EXCLUDE_FROM_ALL'
b='third_party/foxi-src EXCLUDE_FROM_ALL'
print('foxi path patched' if a in s else 'MISS')
io.open(p,'w',encoding='utf-8').write(s.replace(a,b))