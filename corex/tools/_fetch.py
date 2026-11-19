import urllib.request, tarfile, io, os, shutil
def get(url, dst, strip):
    print('fetch', url)
    data=urllib.request.urlopen(url, timeout=120).read()
    t=tarfile.open(fileobj=io.BytesIO(data))
    tmp=dst+'.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    t.extractall(tmp)
    root=os.path.join(tmp, strip)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(root, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print('ok', dst, len(os.listdir(dst)))
T='/data/home/2024240384035/build/torch24/third_party'
get('https://codeload.github.com/pytorch/kineto/tar.gz/refs/heads/main', T+'/kineto', 'kineto-main')
get('https://codeload.github.com/onnx/onnx/tar.gz/refs/tags/v1.16.1', T+'/onnx', 'onnx-1.16.1')