#!/bin/bash
sed -n '40,75p' /usr/local/corex/include/thrust/mr/new.h
echo "== CUSPARSE =="
grep -n "CUSPARSE_SPMM" /usr/local/corex/include/cusparse.h | head -12
echo "== CUDA_H =="
grep -rn "maxBlocksPerMultiProcessor" /usr/local/corex/include/driver_types.h /usr/local/corex/include/cuda_runtime_api.h 2>/dev/null | head -3
