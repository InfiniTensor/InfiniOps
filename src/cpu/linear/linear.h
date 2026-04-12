#ifndef INFINI_OPS_CPU_LINEAR_LINEAR_H_
#define INFINI_OPS_CPU_LINEAR_LINEAR_H_

#include <utility>

#include "base/linear.h"
#include "common/generic_utils.h"
#include "cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kCpu> : public Linear,
                                             Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<Tensor> bias,
           bool trans_a, bool trans_b, Tensor out)
      : Linear{a, b, bias, trans_a, trans_b, out} {}

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(a, b, bias, out);
        },
        "`Operator<Linear, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor a, const Tensor b, std::optional<Tensor> bias,
               Tensor out) const {
    const auto* A = static_cast<const T*>(a.data());
    const auto* B = static_cast<const T*>(b.data());
    auto* Out = static_cast<T*>(out.data());
    const T* Bias = bias ? static_cast<const T*>(bias->data()) : nullptr;

    for (Tensor::Size batch = 0; batch < batch_count_; ++batch) {
      const auto* A_batch = A + batch * batch_stride_a_;
      const auto* B_batch = B + batch * batch_stride_b_;
      auto* Out_batch = Out + batch * batch_stride_c_;

      for (Tensor::Size i = 0; i < m_; ++i) {

        for (Tensor::Size j = 0; j < n_; ++j) {
          float sum = 0.0f;

          for (Tensor::Size l = 0; l < k_; ++l) {
            float a_val = Cast<float>(
                A_batch[trans_a_ ? (l * lda_ + i) : (i * lda_ + l)]);
            float b_val = Cast<float>(
                B_batch[trans_b_ ? (j * ldb_ + l) : (l * ldb_ + j)]);
            sum += a_val * b_val;
          }

          if (Bias) {
            sum += Cast<float>(Bias[j]);
          }

          Out_batch[i * ldc_ + j] = Cast<T>(sum);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
