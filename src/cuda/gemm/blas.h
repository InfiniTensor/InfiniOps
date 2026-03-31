#ifndef INFINI_OPS_CUDA_GEMM_BLAS_H_
#define INFINI_OPS_CUDA_GEMM_BLAS_H_

#include <utility>

#include "base/gemm.h"

namespace infini::ops {

template <typename Backend>
class Blas : public Gemm {
 public:
  Blas(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, std::optional<int> trans_a,
       std::optional<int> trans_b, Tensor c)
      : Gemm{a, b, alpha, beta, trans_a, trans_b, c},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{c.stride(-1) == 1} {
    // TODO: Check constraints.
  }

  Blas(const Tensor a, const Tensor b, std::optional<float> alpha,
       std::optional<float> beta, Tensor c)
      : Blas{a, b, alpha, beta, std::nullopt, std::nullopt, c} {}

  Blas(const Tensor a, const Tensor b, Tensor c)
      : Blas{a, b, std::nullopt, std::nullopt, std::nullopt, std::nullopt, c} {}

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    Backend::blasSetStream(GetHandle(),
                           static_cast<typename Backend::stream_t>(stream_));

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto& beta_value{beta.value_or(beta_)};

    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};
    auto op_a{GetOpA(trans_a_value, trans_b_value)};
    auto op_b{GetOpB(trans_a_value, trans_b_value)};
    const void* alpha_ptr{GetAlphaPtr(alpha_value, c.dtype())};
    const void* beta_ptr{GetBetaPtr(beta_value, c.dtype())};

    Backend::blasGemmStridedBatchedEx(
        GetHandle(), op_a, op_b, swap_a_and_b_ ? n_ : m_,
        swap_a_and_b_ ? m_ : n_, k_, alpha_ptr,
        swap_a_and_b_ ? b.data() : a.data(),
        Backend::GetDataType(swap_a_and_b_ ? b.dtype() : a.dtype()),
        swap_a_and_b_ ? ldb_ : lda_,
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_,
        swap_a_and_b_ ? a.data() : b.data(),
        Backend::GetDataType(swap_a_and_b_ ? a.dtype() : b.dtype()),
        swap_a_and_b_ ? lda_ : ldb_,
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_, beta_ptr, c.data(),
        Backend::GetDataType(c.dtype()), ldc_, batch_stride_c_, batch_count_,
        Backend::GetComputeType(c.dtype()), Backend::BLAS_GEMM_DEFAULT);
  }

 protected:
  virtual const void* GetAlphaPtr(const float& alpha, DataType) const {
    return &alpha;
  }

  virtual const void* GetBetaPtr(const float& beta, DataType) const {
    return &beta;
  }

 private:
  auto GetOpA(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }
    return (a_is_col_major_ != trans_a) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

  auto GetOpB(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }
    return (b_is_col_major_ != trans_b) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

  static typename Backend::blasHandle_t& GetHandle() {
    static typename Backend::blasHandle_t handle = []() {
      typename Backend::blasHandle_t h;
      Backend::blasCreate(&h);
      return h;
    }();
    return handle;
  }

  bool a_is_col_major_{false};

  bool b_is_col_major_{false};

  bool swap_a_and_b_{false};
};

}  // namespace infini::ops

#endif
