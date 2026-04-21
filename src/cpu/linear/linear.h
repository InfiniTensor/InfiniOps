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
          Compute<T>(a, b, bias, trans_a, trans_b, out);
        },
        "`Operator<Linear, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor a, const Tensor b, std::optional<Tensor> bias,
               bool trans_a, bool trans_b, Tensor out) const {
    const auto* A = static_cast<const T*>(a.data());
    const auto* B = static_cast<const T*>(b.data());
    auto* Out = static_cast<T*>(out.data());
    const T* Bias = bias ? static_cast<const T*>(bias->data()) : nullptr;

    // Determine M, K, N from shapes and transpose flags.
    auto ndim_a = a_shape_.size();
    auto ndim_b = b_shape_.size();
    auto ndim_out = out_shape_.size();

    Tensor::Size M = out_shape_[ndim_out - 2];
    Tensor::Size N = out_shape_[ndim_out - 1];
    Tensor::Size K = trans_a ? a_shape_[ndim_a - 2] : a_shape_[ndim_a - 1];

    // Compute strides for the inner matrix dimensions after transpose.
    Tensor::Stride stride_a_m =
        trans_a ? a_strides_[ndim_a - 1] : a_strides_[ndim_a - 2];
    Tensor::Stride stride_a_k =
        trans_a ? a_strides_[ndim_a - 2] : a_strides_[ndim_a - 1];
    Tensor::Stride stride_b_k =
        trans_b ? b_strides_[ndim_b - 1] : b_strides_[ndim_b - 2];
    Tensor::Stride stride_b_n =
        trans_b ? b_strides_[ndim_b - 2] : b_strides_[ndim_b - 1];
    Tensor::Stride stride_out_m = out_strides_[ndim_out - 2];
    Tensor::Stride stride_out_n = out_strides_[ndim_out - 1];

    // Batch dimensions.
    Tensor::Size batch_count = 1;
    for (size_t i = 0; i + 2 < ndim_out; ++i) {
      batch_count *= out_shape_[i];
    }

    Tensor::Stride batch_stride_a = ndim_a > 2 ? a_strides_[ndim_a - 3] : 0;
    Tensor::Stride batch_stride_b = ndim_b > 2 ? b_strides_[ndim_b - 3] : 0;
    Tensor::Stride batch_stride_out =
        ndim_out > 2 ? out_strides_[ndim_out - 3] : 0;

    // Bias stride: for 1D bias [N], stride is 1. For batched bias, use last
    // stride.
    Tensor::Stride bias_stride = 0;
    if (Bias && bias) {
      auto ndim_bias = bias->shape().size();
      bias_stride = bias->strides()[ndim_bias - 1];
    }

    for (Tensor::Size batch = 0; batch < batch_count; ++batch) {
      const auto* A_batch = A + batch * batch_stride_a;
      const auto* B_batch = B + batch * batch_stride_b;
      auto* Out_batch = Out + batch * batch_stride_out;

      for (Tensor::Size i = 0; i < M; ++i) {
        for (Tensor::Size j = 0; j < N; ++j) {
          float sum = 0.0f;

          for (Tensor::Size l = 0; l < K; ++l) {
            float a_val = Cast<float>(A_batch[i * stride_a_m + l * stride_a_k]);
            float b_val = Cast<float>(B_batch[l * stride_b_k + j * stride_b_n]);
            sum += a_val * b_val;
          }

          if (Bias) {
            sum += Cast<float>(Bias[j * bias_stride]);
          }

          Out_batch[i * stride_out_m + j * stride_out_n] = Cast<T>(sum);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
