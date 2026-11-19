import io, os, shutil, urllib.request, tarfile
R='/data/home/2024240384035/build/torch24'

# 1 sparse default alg
p=R+'/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu'
s=io.open(p,encoding='utf-8').read()
if 'CUSPARSE_SPMM_ALG_DEFAULT' in s and '#define CUSPARSE_SPMM_ALG_DEFAULT' not in s:
    a='#ifndef CUSPARSE_SPMM_COO_ALG2\n#define CUSPARSE_SPMM_COO_ALG2 ((cusparseSpMMAlg_t)2)\n#endif'
    b=a+'\n#ifndef CUSPARSE_SPMM_ALG_DEFAULT\n#define CUSPARSE_SPMM_ALG_DEFAULT ((cusparseSpMMAlg_t)0)\n#endif'
    if a in s:
        io.open(p,'w',encoding='utf-8').write(s.replace(a,b,1)); print('spmm default added')
    else: print('spmm anchor miss')
else: print('spmm default ok')

# 2 compat opaque struct
h=R+'/c10/cuda/CoreXCompat.h'
s=io.open(h,encoding='utf-8').read()
if 'struct cublasLtMatmulDescOpaque_t' not in s:
    s=s.replace('#if __has_include(<cublasLt.h>)','#if !__has_include(<cublasLt.h>)\nstruct cublasLtMatmulDescOpaque_t;\n#endif\n#if __has_include(<cublasLt.h>)')
    io.open(h,'w',encoding='utf-8').write(s); print('opaque struct added')
else: print('opaque ok')

# 3 kineto v0.4.0 fresh dir
T=R+'/third_party'
data=urllib.request.urlopen('https://codeload.github.com/pytorch/kineto/tar.gz/refs/tags/v0.4.0', timeout=120).read()
tmp=T+'/kineto-v04.tmp'; shutil.rmtree(tmp, ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(data)).extractall(tmp)
root=os.path.join(tmp,'kineto-0.4.0')
dst=T+'/kineto-v04'
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(root,dst)
shutil.rmtree(tmp)
print('kineto v04', os.path.exists(dst+'/libkineto/include/ActivityType.h'))
p=R+'/cmake/Dependencies.cmake'
s=io.open(p,encoding='utf-8').read()
s=s.replace('kineto-src/libkineto/include','kineto-v04/libkineto/include')
io.open(p,'w',encoding='utf-8').write(s); print('kineto path v04 patched')