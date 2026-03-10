#ifndef INFINI_OPS_MOORE_GEMM_MUBLAS_H_
#define INFINI_OPS_MOORE_GEMM_MUBLAS_H_

#include <mublas.h>
#include <musa_runtime_api.h>

#include "base/gemm.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kMoore> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{c.stride(-1) == 1} {
    mublasCreate(&handle_);
  }

  ~Operator() { mublasDestroy(handle_); }

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 c} {}

  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, Tensor c)
      : Operator{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    mublasSetStream(handle_, static_cast<musaStream_t>(stream_));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};
    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};

    mublasOperation_t op_a = GetOpA(trans_a_value, trans_b_value);
    mublasOperation_t op_b = GetOpB(trans_a_value, trans_b_value);

    const int m = static_cast<int>(swap_a_and_b_ ? n_ : m_);
    const int n = static_cast<int>(swap_a_and_b_ ? m_ : n_);
    const int k = static_cast<int>(k_);
    const void* a_ptr = swap_a_and_b_ ? b.data() : a.data();
    const void* b_ptr = swap_a_and_b_ ? a.data() : b.data();
    const int lda = static_cast<int>(swap_a_and_b_ ? ldb_ : lda_);
    const int ldb = static_cast<int>(swap_a_and_b_ ? lda_ : ldb_);
    const int ldc = static_cast<int>(ldc_);
    const int64_t stride_a =
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_;
    const int64_t stride_b =
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_;
    const int64_t stride_c = batch_stride_c_;
    const int batch = static_cast<int>(batch_count_);
    const auto a_type = GetDataType(swap_a_and_b_ ? b.dtype() : a.dtype());
    const auto b_type = GetDataType(swap_a_and_b_ ? a.dtype() : b.dtype());
    const auto c_type = GetDataType(c.dtype());
    const auto compute_type = GetComputeType(c.dtype());

    Float16 alpha_fp16{};
    Float16 beta_fp16{};
    const void* alpha_ptr = &alpha_value;
    const void* beta_ptr = &beta_value;

    // MUSA GEMM requires alpha/beta to use the same dtype as compute_type.
    if (compute_type == MUBLAS_COMPUTE_16F) {
      alpha_fp16 = Float16::FromFloat(alpha_value);
      beta_fp16 = Float16::FromFloat(beta_value);
      alpha_ptr = &alpha_fp16;
      beta_ptr = &beta_fp16;
    }

    mublasGemmStridedBatchedEx(
        handle_, op_a, op_b, m, n, k, alpha_ptr, a_ptr, a_type, lda, stride_a,
        b_ptr, b_type, ldb, stride_b, beta_ptr, c.data(), c_type,
        ldc, stride_c, batch, compute_type,
        MUBLAS_GEMM_DEFAULT);
  }

 private:
  static musaDataType_t GetDataType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MUSA_R_16F;
    if (dtype == DataType::kBFloat16) return MUSA_R_16BF;
    return MUSA_R_32F;
  }

  static mublasComputeType_t GetComputeType(DataType dtype) {
    if (dtype == DataType::kFloat16) return MUBLAS_COMPUTE_16F;
    if (dtype == DataType::kBFloat16) return MUBLAS_COMPUTE_32F;
    return MUBLAS_COMPUTE_32F;
  }

  mublasOperation_t GetOpA(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? MUBLAS_OP_T : MUBLAS_OP_N;
    }
    return (a_is_col_major_ != trans_a) ? MUBLAS_OP_T : MUBLAS_OP_N;
  }

  mublasOperation_t GetOpB(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? MUBLAS_OP_T : MUBLAS_OP_N;
    }
    return (b_is_col_major_ != trans_b) ? MUBLAS_OP_T : MUBLAS_OP_N;
  }

  bool a_is_col_major_{false};
  bool b_is_col_major_{false};
  bool swap_a_and_b_{false};
  mublasHandle_t handle_;
};

}  // namespace infini::ops

#endif
