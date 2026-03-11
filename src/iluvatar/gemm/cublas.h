#ifndef INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_
#define INFINI_OPS_ILUVATAR_GEMM_CUBLAS_H_

#include <utility>

// clang-format off
#include "cublas_v2.h"
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

namespace gemm {

struct IluvatarBackend {
  using blasHandle_t = cublasHandle_t;

  using stream_t = cudaStream_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;

  static constexpr auto R_16F = CUDA_R_16F;

  static constexpr auto R_16BF = CUDA_R_16BF;

  static constexpr auto R_32F = CUDA_R_32F;

  // Iluvatar uses `cudaDataType` for `computeType`, so we need to use
  // `CUDA_R_32F` instead of `CUBLAS_COMPUTE_32F_FAST_TF32`.
  static constexpr auto BLAS_COMPUTE_32F = CUDA_R_32F;
  
  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = CUDA_R_32F;

  // Iluvatar uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` instead of
  // `CUBLAS_GEMM_DEFAULT`.
  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  static constexpr auto blasCreate = cublasCreate;

  static constexpr auto blasSetStream = cublasSetStream;

  static constexpr auto blasDestroy = cublasDestroy;

  static constexpr auto blasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };

  static auto GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return R_16F;
    if (dtype == DataType::kBFloat16) return R_16BF;
    return R_32F;
  }

  static auto GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
      return BLAS_COMPUTE_32F;
    return BLAS_COMPUTE_32F_FAST_TF32;
  }
};

}  // namespace gemm

template <>
class Operator<Gemm, Device::Type::kIluvatar>
    : public Blas<gemm::IluvatarBackend> {
 public:
  using Blas<gemm::IluvatarBackend>::Blas;
};

}  // namespace infini::ops

#endif
