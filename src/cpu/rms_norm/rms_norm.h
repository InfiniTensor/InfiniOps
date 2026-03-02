#ifndef INFINI_OPS_CPU_RMS_NORM_H_
#define INFINI_OPS_CPU_RMS_NORM_H_

#include <cmath>

#include "base/rms_norm.h"
#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kCpu> : public RmsNorm {
 public:
  Operator(const Tensor y, const Tensor x, const Tensor w, float epsilon)
      : RmsNorm{y, x, w, epsilon} {}

  Operator(const Tensor y, const Tensor x, const Tensor w)
      : Operator{y, x, w, 1e-6f} {}

  void operator()(void* stream, Tensor y, const Tensor x, const Tensor w,
                  float /*epsilon*/ = 0) const override {
    // CPU backend supports fp32 only; fp16/bf16 use GPU backends.
    if (y.dtype() != DataType::kFloat32 || x.dtype() != DataType::kFloat32 ||
        w.dtype() != DataType::kFloat32) {
      abort();
    }

    auto* y_ptr = static_cast<float*>(y.data());
    const auto* x_ptr = static_cast<const float*>(x.data());
    const auto* w_ptr = static_cast<const float*>(w.data());

    auto stride_x_batch = x_strides_.size() > 1 ? x_strides_[0] : 0;
    auto stride_x_nhead = x_strides_.size() > 1 ? x_strides_[1] : x_strides_[0];
    auto stride_y_batch = y_strides_.size() > 1 ? y_strides_[0] : 0;
    auto stride_y_nhead = y_strides_.size() > 1 ? y_strides_[1] : y_strides_[0];

    for (Tensor::Size bi = 0; bi < batch_size_; ++bi) {
      for (Tensor::Size hi = 0; hi < nhead_; ++hi) {
        const float* x_row = x_ptr + bi * stride_x_batch + hi * stride_x_nhead;
        float* y_row = y_ptr + bi * stride_y_batch + hi * stride_y_nhead;

        float ss = 0;
        for (Tensor::Size k = 0; k < dim_; ++k) {
          float v = x_row[k];
          ss += v * v;
        }
        float rms = 1.f / std::sqrt(ss / static_cast<float>(dim_) + epsilon_);

        for (Tensor::Size k = 0; k < dim_; ++k) {
          y_row[k] = x_row[k] * w_ptr[k] * rms;
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
