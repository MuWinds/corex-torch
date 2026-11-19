#pragma once
#if defined(__ILUVATAR__)
#include <cublas_v2.h>
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#ifndef CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
#define CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION 0
#endif
#endif
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
struct cublasLtMatmulDescOpaque_t;
typedef cudaDataType_t cublasComputeType_t;
#ifndef CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_64F CUDA_R_64F
#define CUBLAS_COMPUTE_32F_FAST_TF32 CUDA_R_32F
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#endif
#endif
