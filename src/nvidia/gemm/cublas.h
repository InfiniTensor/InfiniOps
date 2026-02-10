#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

struct CudaBackend {
  using blasHandle_t = cublasHandle_t;
  using stream_t = cudaStream_t;

  static void blasCreate(blasHandle_t* handle) { cublasCreate(handle); }

  static void blasSetStream(blasHandle_t handle, stream_t stream) {
    cublasSetStream(handle, stream);
  }

  static void blasGemmEx(blasHandle_t handle, bool transA, bool transB, int m,
                         int n, int k, const float* alpha, const void* B,
                         int ldb, const void* A, int lda, const float* beta,
                         void* C, int ldc) {
    cublasGemmEx(handle, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                 transB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, alpha, B,
                 CUDA_R_32F, ldb, A, CUDA_R_32F, lda, beta, C, CUDA_R_32F, ldc,
                 CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
  }

  static void blasDestroy(blasHandle_t handle) { cublasDestroy(handle); }
};

template <>
class Operator<Gemm, Device::Type::kNvidia> : public Blas<CudaBackend> {
 public:
  using Blas<CudaBackend>::Blas;
};

}  // namespace infini::ops

#endif
