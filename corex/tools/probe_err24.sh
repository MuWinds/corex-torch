#!/bin/bash
T=~/build/torch24
echo "==CVTA=="
sed -n '95,120p' $T/third_party/cutlass/include/cute/arch/util.hpp
echo "==VARIADIC=="
sed -n '462,482p' $T/third_party/cutlass/include/cutlass/functional.h
echo "==BFE_SRC=="
grep -rn "bfe.u32" $T/aten $T/third_party/cutlass/include 2>/dev/null | head -5
echo "==ONNX_HEAD=="
sed -n '1,12p' $T/torch/csrc/jit/serialization/onnx.cpp
grep -n "USE_ONNX\|onnx" $T/torch/CMakeLists.txt | head -8
echo "==CUSPARSE=="
grep -n "CUSPARSE_SPMM.*ALG" /usr/local/corex/include/cusparse.h 2>/dev/null | head -8
echo "==THRUST=="
sed -n '55,75p' /usr/local/corex/include/thrust/mr/new.h 2>/dev/null
echo "==THRUST_GUARD=="
grep -n "__cpp_aligned_new\|align_val_t" /usr/local/corex/include/thrust/mr/new.h 2>/dev/null | head -6
echo "==TOPK=="
sed -n '612,628p' $T/aten/src/ATen/native/cuda/TensorTopK.cu
echo "==DONE=="
