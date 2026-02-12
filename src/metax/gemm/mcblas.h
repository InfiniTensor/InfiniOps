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

  static constexpr auto BLAS_OP_N = MCBLAS_OP_N;
  static constexpr auto BLAS_OP_T = MCBLAS_OP_T;
  static constexpr auto R_32F = MACA_R_32F;
  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
      MCBLAS_COMPUTE_32F_FAST_TF32;
  static constexpr auto BLAS_GEMM_DEFAULT = MCBLAS_GEMM_DEFAULT;

  static constexpr auto blasCreate = mcblasCreate;
  static constexpr auto blasSetStream = mcblasSetStream;
  static constexpr auto blasDestroy = mcblasDestroy;

  static constexpr mcblasStatus_t (*blasGemmEx)(
      mcblasHandle_t, mcblasOperation_t, mcblasOperation_t, int, int, int,
      const void*, const void*, macaDataType_t, int, const void*,
      macaDataType_t, int, const void*, void*, macaDataType_t, int,
      mcblasComputeType_t, mcblasGemmAlgo_t) = mcblasGemmEx;
};

template <>
class Operator<Gemm, Device::Type::kMetax> : public Blas<MetaxBackend> {
 public:
  using Blas<MetaxBackend>::Blas;
};

}  // namespace infini::ops

#endif
