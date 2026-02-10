#ifndef INFINI_OPS_METAX_GEMM_MCBLAS_H_
#define INFINI_OPS_METAX_GEMM_MCBLAS_H_

#include <utility>

// clang-format off
#include <mcblas/mcblas.h>
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

struct MetaxBackend {
  using blasHandle_t = mcblasHandle_t;
  using stream_t = mcStream_t;

  static void blasCreate(blasHandle_t* handle) { mcblasCreate(handle); }

  static void blasSetStream(blasHandle_t handle, stream_t stream) {
    mcblasSetStream(handle, stream);
  }

  static void blasGemmEx(blasHandle_t handle, bool transA, bool transB, int m,
                         int n, int k, const float* alpha, const void* B,
                         int ldb, const void* A, int lda, const float* beta,
                         void* C, int ldc) {
    mcblasGemmEx(handle, transA ? MCBLAS_OP_T : MCBLAS_OP_N,
                 transB ? MCBLAS_OP_T : MCBLAS_OP_N, m, n, k, alpha, B,
                 MACA_R_32F, ldb, A, MACA_R_32F, lda, beta, C, MACA_R_32F, ldc,
                 MCBLAS_COMPUTE_32F_FAST_TF32, MCBLAS_GEMM_DEFAULT);
  }

  static void blasDestroy(blasHandle_t handle) { mcblasDestroy(handle); }
};

template <>
class Operator<Gemm, Device::Type::kMetax> : public Blas<MetaxBackend> {
 public:
  using Blas<MetaxBackend>::Blas;
};

}  // namespace infini::ops

#endif
