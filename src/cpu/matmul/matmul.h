#ifndef INFINI_OPS_CPU_MATMUL_H_
#define INFINI_OPS_CPU_MATMUL_H_

#include <utility>

#include "base/matmul.h"
#include "common/generic_utils.h"
#include "cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kCpu> : public Matmul,
                                             Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c, bool trans_a,
           bool trans_b)
      : Matmul{a, b, c, trans_a, trans_b} {
    // TODO: Check constraints.
  }

  Operator(const Tensor a, const Tensor b, Tensor c)
      : Operator{a, b, c, false, false} {}

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        c.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(a, b, c, trans_a, trans_b);
        },
        "`Operator<Matmul, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor a, const Tensor b, Tensor c, bool trans_a,
               bool trans_b) const {
    const auto* A = static_cast<const T*>(a.data());
    const auto* B = static_cast<const T*>(b.data());
    auto* C = static_cast<T*>(c.data());

    Tensor::Stride stride_a_m = trans_a
                                    ? a_strides_[a_strides_.size() - 1]
                                    : a_strides_[a_strides_.size() - 2];
    Tensor::Stride stride_a_k = trans_a
                                    ? a_strides_[a_strides_.size() - 2]
                                    : a_strides_[a_strides_.size() - 1];
    Tensor::Stride stride_b_k = trans_b
                                    ? b_strides_[b_strides_.size() - 1]
                                    : b_strides_[b_strides_.size() - 2];
    Tensor::Stride stride_b_n = trans_b
                                    ? b_strides_[b_strides_.size() - 2]
                                    : b_strides_[b_strides_.size() - 1];
    Tensor::Stride stride_c_m = c_strides_[c_strides_.size() - 2];
    Tensor::Stride stride_c_n = c_strides_[c_strides_.size() - 1];

    for (Tensor::Size batch = 0; batch < batch_count_; ++batch) {
      const auto* A_batch = A + batch * batch_stride_a_;
      const auto* B_batch = B + batch * batch_stride_b_;
      auto* C_batch = C + batch * batch_stride_c_;

      for (Tensor::Size i = 0; i < m_; ++i) {
        for (Tensor::Size j = 0; j < n_; ++j) {
          float sum = 0.0f;

          for (Tensor::Size l = 0; l < k_; ++l) {
            float a_val = Cast<float>(A_batch[i * stride_a_m + l * stride_a_k]);
            float b_val = Cast<float>(B_batch[l * stride_b_k + j * stride_b_n]);
            sum += a_val * b_val;
          }

          Tensor::Size idx = i * stride_c_m + j * stride_c_n;
          C_batch[idx] = Cast<T>(sum);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
