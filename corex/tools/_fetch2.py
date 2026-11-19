import urllib.request, tarfile, io, os, shutil
def get(url, dst, strip):
    data=urllib.request.urlopen(url, timeout=120).read()
    t=tarfile.open(fileobj=io.BytesIO(data))
    tmp=dst+'.tmp'; shutil.rmtree(tmp, ignore_errors=True); t.extractall(tmp)
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(os.path.join(tmp,strip), dst)
    shutil.rmtree(tmp, ignore_errors=True); print('ok',dst)
T='/data/home/2024240384035/build/torch24/third_party'
get('https://codeload.github.com/onnx/foxi/tar.gz/refs/heads/master', T+'/foxi', 'foxi-master')
print(os.path.exists(T+'/foxi/CMakeLists.txt'))