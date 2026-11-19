#!/bin/bash
# Round-2 CoreX/torch2.4 compatibility fixes. Idempotent.
set -u
T=$HOME/build/torch24
echo "=== [1] EmbeddingBag bfe: real bit ops"

python3 - <<'PY'
import io
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/quantized/cuda/EmbeddingBag.cu'
s=io.open(p,encoding='utf-8').read()
old='''#if defined(USE_ROCM) || defined(__ILUVATAR__)
  return *reinterpret_cast<float*>((val >> pos) && ((1u << len) - 1u ));
#else'''
new='''#if defined(USE_ROCM) || defined(__ILUVATAR__)
  uint32_t v = (val >> pos) & (len >= 32u ? 0xFFFFFFFFu : ((1u << len) - 1u));
  return __uint2float_rn(v);
#else'''
if old in s:
    s=s.replace(old,new)
    io.open(p,'w',encoding='utf-8').write(s)
    print('bfe fixed')
else:
    print('bfe already ok' if 'uint32_t v = (val >> pos)' in s else 'bfe PATTERN MISS')
PY

echo "=== [2] cutlass functional.h: second functor non-variadic"
python3 - <<'PY'
import io
p='/data/home/2024240384035/build/torch24/third_party/cutlass/include/cutlass/functional.h'
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
    s=s.replace(old,new)
    io.open(p,'w',encoding='utf-8').write(s)
    print('second fixed')
else:
    print('second ok' if 'struct second {\n#if defined(__ILUVATAR__)' in s else 'second PATTERN MISS')
PY

echo "=== [3] cusparse SpMM algo enums fallback"
python3 - <<'PY'
import io,re
p='/data/home/2024240384035/build/torch24/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu'
s=io.open(p,encoding='utf-8').read()
if 'CoreX: cusparse lacks SPMM_COO_ALG' not in s:
    anchor='namespace at {\nnamespace native {\n'
    block='''#if defined(__ILUVATAR__)
// CoreX: cusparse lacks SPMM_COO_ALG
#ifndef CUSPARSE_SPMM_COO_ALG1
#define CUSPARSE_SPMM_COO_ALG1 1
#endif
#ifndef CUSPARSE_SPMM_COO_ALG2
#define CUSPARSE_SPMM_COO_ALG2 2
#endif
#endif
'''
    if anchor in s:
        s=s.replace(anchor,anchor+block,1)
        io.open(p,'w',encoding='utf-8').write(s)
        print('cusparse enums added')
    else:
        print('cusparse ANCHOR MISS')
else:
    print('cusparse enums already')
PY

echo "=== [4] -fsized-deallocation (thrust mr/new.h)"
python3 - <<'PY'
import io
p='/data/home/2024240384035/build/torch24/cmake/public/cuda.cmake'
s=io.open(p,encoding='utf-8').read()
if 'fsized-deallocation' not in s:
    s=s.replace('add_definitions(-D__ILUVATAR__)',
                'add_compile_options(-fsized-deallocation)\nadd_definitions(-D__ILUVATAR__)',1)
    io.open(p,'w',encoding='utf-8').write(s)
    print('sized-dealloc added')
else:
    print('sized-dealloc already')
PY

echo "=== [5] scan script env (keep USE_CUDA=1 on reconfigure)"
cat > $HOME/build/scan24_clean.sh <<'EOS'
#!/bin/bash
set -x
export USE_CUDA=1 USE_CUDNN=0 USE_KINETO=0 USE_NCCL=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 \
 USE_GLOO=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_NNPACK=0 USE_XNNPACK=0 \
 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_FBGEMM=0 USE_MKLDNN=0 USE_ONNX=0 USE_ITT=0 \
 BUILD_TEST=0 SKIP_SUBMODULE_CHECK=1 MAX_JOBS=48 CUDA_HOME=/usr/local/corex
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24
cmake . > $HOME/build/reconf24c.log 2>&1 || { echo RECONF_FAIL; exit 1; }
cd build
ninja -k 50 > $HOME/build/scan24c.log 2>&1
echo "RC=$?"
echo "FAIL=$(grep -c 'FAILED:' $HOME/build/scan24c.log)"
echo "CUOBJ=$(grep -c '\.cu\.o' build.ninja)"
EOS
chmod +x $HOME/build/scan24_clean.sh
echo DONE
