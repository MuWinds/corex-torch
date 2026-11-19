import io, base64
R='/data/home/2024240384035/build/torch24'

p=R+'/aten/src/ATen/native/quantized/cuda/EmbeddingBag.cu'
s=io.open(p,encoding='utf-8').read()
old='''#if defined(USE_ROCM) || defined(__ILUVATAR__)
  return *reinterpret_cast<float*>((val >> pos) && ((1u << len) - 1u ));
#else'''
new='''#if defined(USE_ROCM) || defined(__ILUVATAR__)
  uint32_t v = (val >> pos) & (len >= 32u ? 0xFFFFFFFFu : ((1u << len) - 1u));
  return __uint2float_rn(v);
#else'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new)); print('bfe fixed')
else:
    print('bfe ok' if 'uint32_t v = (val >> pos)' in s else 'bfe MISS')

p=R+'/third_party/cutlass/include/cutlass/functional.h'
s=io.open(p,encoding='utf-8').read()
old='''template <typename T>
struct second {
  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {'''
new='''template <typename T>
struct second {
#if defined(__ILUVATAR__)
  CUTLASS_HOST_DEVICE
  T operator()(T const &, T const & second) const {
    return second;
  }
  CUTLASS_HOST_DEVICE
  T operator()(T const & first) const {
    return first;
  }
#else
  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {'''
if old in s:
    io.open(p,'w',encoding='utf-8').write(s.replace(old,new)); print('second fixed')
else:
    print('second ok' if 'struct second {\n#if defined(__ILUVATAR__)' in s else 'second MISS')

p=R+'/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu'
s=io.open(p,encoding='utf-8').read()
if 'CoreX: cusparse lacks SPMM_COO_ALG' not in s:
    a='namespace at {\nnamespace native {\n'
    b='''#if defined(__ILUVATAR__)
// CoreX: cusparse lacks SPMM_COO_ALG
#ifndef CUSPARSE_SPMM_COO_ALG1
#define CUSPARSE_SPMM_COO_ALG1 1
#endif
#ifndef CUSPARSE_SPMM_COO_ALG2
#define CUSPARSE_SPMM_COO_ALG2 2
#endif
#endif
'''
    if a in s:
        io.open(p,'w',encoding='utf-8').write(s.replace(a,a+b,1)); print('spmm added')
    else:
        print('spmm ANCHOR MISS')
else:
    print('spmm already')

p=R+'/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
if 'fsized-deallocation' not in s:
    io.open(p,'w',encoding='utf-8').write(s.replace('add_definitions(-D__ILUVATAR__)','add_compile_options(-fsized-deallocation)\nadd_definitions(-D__ILUVATAR__)',1)); print('sized-dealloc added')
else:
    print('sized-dealloc already')
