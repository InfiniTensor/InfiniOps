#ifndef INFINI_OPS_METAX_GEMM_MCBLAS_H_
#define INFINI_OPS_METAX_GEMM_MCBLAS_H_

#include <utility>

// clang-format off
#include <mcblas/mcblas.h>
// clang-format on

#include "cuda/gemm/blas.h"

namespace infini::ops {

namespace gemm {

struct MetaxBackend {
  using blasHandle_t = mcblasHandle_t;

  using stream_t = mcStream_t;

  static constexpr auto BLAS_OP_N = MCBLAS_OP_N;

  static constexpr auto BLAS_OP_T = MCBLAS_OP_T;

  static constexpr auto R_16F = MACA_R_16F;

  static constexpr auto R_16BF = MACA_R_16BF;

  static constexpr auto R_32F = MACA_R_32F;

  static constexpr auto BLAS_COMPUTE_32F = MCBLAS_COMPUTE_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      MCBLAS_COMPUTE_32F_FAST_TF32;

  static constexpr auto BLAS_GEMM_DEFAULT = MCBLAS_GEMM_DEFAULT;

  static constexpr auto blasCreate = mcblasCreate;

  static constexpr auto blasSetStream = mcblasSetStream;

  static constexpr auto blasDestroy = mcblasDestroy;

  static constexpr auto blasGemmStridedBatchedEx = [](auto&&... args) {
    return mcblasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
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
class Operator<Gemm, Device::Type::kMetax> : public Blas<gemm::MetaxBackend> {
 public:
  using Blas<gemm::MetaxBackend>::Blas;
};

}  // namespace infini::ops

#endif
