#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

struct NvidiaBackend {
  using blasHandle_t = cublasHandle_t;
  using stream_t = cudaStream_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;
  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;
  static constexpr auto R_32F = CUDA_R_32F;
  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      CUBLAS_COMPUTE_32F_FAST_TF32;
  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  static constexpr auto blasCreate = cublasCreate;
  static constexpr auto blasSetStream = cublasSetStream;
  static constexpr auto blasDestroy = cublasDestroy;

  static constexpr auto blasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };
};

template <>
class Operator<Gemm, Device::Type::kNvidia> : public Blas<NvidiaBackend> {
 public:
  using Blas<NvidiaBackend>::Blas;
};

}  // namespace infini::ops

#endif
