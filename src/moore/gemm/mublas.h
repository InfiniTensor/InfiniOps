#ifndef INFINI_OPS_MOORE_GEMM_MUBLAS_H_
#define INFINI_OPS_MOORE_GEMM_MUBLAS_H_

#include <mublas.h>
#include <musa_runtime_api.h>

#include <utility>

#include "cuda/gemm/blas.h"

namespace infini::ops {

namespace gemm {

struct MooreBackend {
  using blasHandle_t = mublasHandle_t;

  using stream_t = musaStream_t;

  static constexpr auto BLAS_OP_N = MUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = MUBLAS_OP_T;

  static constexpr auto R_16F = MUSA_R_16F;

  static constexpr auto R_16BF = MUSA_R_16BF;

  static constexpr auto R_32F = MUSA_R_32F;

  static constexpr auto BLAS_GEMM_DEFAULT = MUBLAS_GEMM_DEFAULT;

  static constexpr auto blasCreate = mublasCreate;

  static constexpr auto blasSetStream = mublasSetStream;

  static constexpr auto blasDestroy = mublasDestroy;

  static constexpr auto blasGemmStridedBatchedEx = [](auto&&... args) {
    return mublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };

  static musaDataType_t GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return R_16F;
    if (dtype == DataType::kBFloat16) return R_16BF;
    return R_32F;
  }

  static mublasComputeType_t GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MUBLAS_COMPUTE_16F;
    if (dtype == DataType::kBFloat16) return MUBLAS_COMPUTE_32F;
    return MUBLAS_COMPUTE_32F;
  }
};

}  // namespace gemm

template <>
class Operator<Gemm, Device::Type::kMoore> : public Blas<gemm::MooreBackend> {
 public:
  using Blas<gemm::MooreBackend>::Blas;

 protected:
  const void* GetAlphaPtr(const float& alpha, DataType dtype) const override {
    if (gemm::MooreBackend::GetComputeType(dtype) == MUBLAS_COMPUTE_16F) {
      alpha_fp16_ = Float16::FromFloat(alpha);
      return &alpha_fp16_;
    }

    return &alpha;
  }

  const void* GetBetaPtr(const float& beta, DataType dtype) const override {
    if (gemm::MooreBackend::GetComputeType(dtype) == MUBLAS_COMPUTE_16F) {
      beta_fp16_ = Float16::FromFloat(beta);
      return &beta_fp16_;
    }

    return &beta;
  }

 private:
  mutable Float16 alpha_fp16_{};

  mutable Float16 beta_fp16_{};
};

}  // namespace infini::ops

#endif
