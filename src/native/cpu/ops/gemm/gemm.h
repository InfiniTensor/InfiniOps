#ifndef INFINI_OPS_CPU_GEMM_H_
#define INFINI_OPS_CPU_GEMM_H_

#include <utility>

#include "base/gemm.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kCpu> : public Gemm,
                                           Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Gemm{a, b, c, alpha, beta, trans_a, trans_b, y} {
    // TODO: Check constraints.
  }

  Operator(const Tensor a, const Tensor b, Tensor y)
      : Operator{a,
                 b,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 y} {}

  using Gemm::operator();

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        y.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(a, b, c, alpha, beta, trans_a, trans_b, y);
        },
        "`Operator<Gemm, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor a, const Tensor b, const std::optional<Tensor> c,
               std::optional<float> alpha, std::optional<float> beta,
               std::optional<int> trans_a, std::optional<int> trans_b,
               Tensor y) const {
    const auto* A = static_cast<const T*>(a.data());
    const auto* B = static_cast<const T*>(b.data());
    const auto* C = c ? static_cast<const T*>(c->data()) : nullptr;
    auto* Y = static_cast<T*>(y.data());

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto beta_value{EffectiveBeta(c, beta)};
    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};
    Tensor::Stride stride_a_m = trans_a_value
                                    ? a_strides_[a_strides_.size() - 1]
                                    : a_strides_[a_strides_.size() - 2];
    Tensor::Stride stride_a_k = trans_a_value
                                    ? a_strides_[a_strides_.size() - 2]
                                    : a_strides_[a_strides_.size() - 1];
    Tensor::Stride stride_b_k = trans_b_value
                                    ? b_strides_[b_strides_.size() - 1]
                                    : b_strides_[b_strides_.size() - 2];
    Tensor::Stride stride_b_n = trans_b_value
                                    ? b_strides_[b_strides_.size() - 2]
                                    : b_strides_[b_strides_.size() - 1];
    const auto y_rank = y_strides_.size();
    const auto stride_c_batch =
        y_rank > 2 ? c_broadcast_strides_[y_rank - 3] : 0;
    const auto stride_c_m = c_broadcast_strides_[y_rank - 2];
    const auto stride_c_n = c_broadcast_strides_[y_rank - 1];
    const auto stride_y_m = y_strides_[y_rank - 2];
    const auto stride_y_n = y_strides_[y_rank - 1];

    for (Tensor::Size b = 0; b < batch_count_; ++b) {
      const auto* A_batch = A + b * batch_stride_a_;
      const auto* B_batch = B + b * batch_stride_b_;
      const auto* C_batch = C ? C + b * stride_c_batch : nullptr;
      auto* Y_batch = Y + b * batch_stride_y_;

      for (Tensor::Size i = 0; i < m_; ++i) {
        for (Tensor::Size j = 0; j < n_; ++j) {
          float sum = 0.0f;

          for (Tensor::Size l = 0; l < k_; ++l) {
            float a_val = Cast<float>(A_batch[i * stride_a_m + l * stride_a_k]);
            float b_val = Cast<float>(B_batch[l * stride_b_k + j * stride_b_n]);
            sum += a_val * b_val;
          }

          const auto c_idx = i * stride_c_m + j * stride_c_n;
          const auto y_idx = i * stride_y_m + j * stride_y_n;
          const float c_val =
              beta_value == 0.0F ? 0.0F : Cast<float>(C_batch[c_idx]);
          Y_batch[y_idx] = Cast<T>(alpha_value * sum + beta_value * c_val);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
