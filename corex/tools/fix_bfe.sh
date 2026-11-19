#!/bin/bash
set -u
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
    print('bfe already ok' if 'uint32_t v = (val >> pos)' in s else 'bfe MISS')
PY
