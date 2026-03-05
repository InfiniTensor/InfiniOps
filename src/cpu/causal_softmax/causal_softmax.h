#ifndef INFINI_OPS_CPU_CAUSAL_SOFTMAX_H_
#define INFINI_OPS_CPU_CAUSAL_SOFTMAX_H_

#include <cmath>

#include "base/causal_softmax.h"
#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kCpu> : public CausalSoftmax {
 public:
  Operator(const Tensor y, const Tensor x) : CausalSoftmax{y, x} {}

  void operator()(void* stream, Tensor y, const Tensor x) const override {
    if (y.dtype() != DataType::kFloat32 || x.dtype() != DataType::kFloat32) {
      std::abort();
    }

    auto* y_ptr = static_cast<float*>(y.data());
    const auto* x_ptr = static_cast<const float*>(x.data());

    auto y_stride_b = ndim_ == 3 ? y_strides_[0] : 0;
    auto y_stride_i = y_strides_[ndim_ - 2];
    auto y_stride_j = y_strides_[ndim_ - 1];
    auto x_stride_b = ndim_ == 3 ? x_strides_[0] : 0;
    auto x_stride_i = x_strides_[ndim_ - 2];
    auto x_stride_j = x_strides_[ndim_ - 1];

    for (Tensor::Size bi = 0; bi < batch_size_; ++bi) {
      for (Tensor::Size i = 0; i < seq_len_; ++i) {
        ptrdiff_t y_offset = bi * y_stride_b + i * y_stride_i;
        ptrdiff_t x_offset = bi * x_stride_b + i * x_stride_i;
        float* y_row = y_ptr + y_offset;
        const float* x_row = x_ptr + x_offset;

        Tensor::Size valid_len = total_seq_len_ - seq_len_ + i + 1;

        for (Tensor::Size j = valid_len; j < total_seq_len_; ++j) {
          y_row[j * y_stride_j] = 0.0f;
        }

        float max_val = x_row[0];
        for (Tensor::Size j = 1; j < valid_len; ++j) {
          float v = x_row[j * x_stride_j];
          if (v > max_val) {
            max_val = v;
          }
        }

        float sum = 0.0f;
        for (Tensor::Size j = 0; j < valid_len; ++j) {
          float v = std::exp(x_row[j * x_stride_j] - max_val);
          y_row[j * y_stride_j] = v;
          sum += v;
        }

        for (Tensor::Size j = 0; j < valid_len; ++j) {
          y_row[j * y_stride_j] /= sum;
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
