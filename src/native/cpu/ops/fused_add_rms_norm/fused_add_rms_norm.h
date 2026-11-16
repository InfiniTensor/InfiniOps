#ifndef INFINI_OPS_CPU_FUSED_ADD_RMS_NORM_H_
#define INFINI_OPS_CPU_FUSED_ADD_RMS_NORM_H_

#include <cmath>
#include <optional>

#include "base/fused_add_rms_norm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cpu/caster_.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kCpu>
    : public FusedAddRmsNorm, Caster<Device::Type::kCpu> {
 public:
  using FusedAddRmsNorm::FusedAddRmsNorm;

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        input.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, residual, weight, epsilon);
        },
        "`Operator<FusedAddRmsNorm, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(Tensor input, Tensor residual,
               const std::optional<Tensor> weight, float epsilon) const {
    auto* input_rows = static_cast<T*>(input.data());
    auto* residual_rows = static_cast<T*>(residual.data());
    const auto* weight_data =
        weight.has_value() ? static_cast<const T*>(weight->data()) : nullptr;
    const auto input_row_stride = input_strides_[input_strides_.size() - 2];
    const auto residual_row_stride =
        residual_strides_[residual_strides_.size() - 2];

#pragma omp parallel for
    for (Tensor::Size row = 0; row < num_tokens_; ++row) {
      T* input_row = input_rows + row * input_row_stride;
      T* residual_row = residual_rows + row * residual_row_stride;

      float sum_squared = 0.0f;
      for (Tensor::Size column = 0; column < dim_; ++column) {
        const float summed =
            Cast<float>(input_row[column]) + Cast<float>(residual_row[column]);
        residual_row[column] = Cast<T>(summed);

        const float stored_sum = Cast<float>(residual_row[column]);
        sum_squared += stored_sum * stored_sum;
      }

      const float reciprocal_rms =
          1.0f / std::sqrt(sum_squared / static_cast<float>(dim_) + epsilon);
      for (Tensor::Size column = 0; column < dim_; ++column) {
        float normalized = Cast<float>(residual_row[column]) * reciprocal_rms;
        if (weight_data != nullptr) {
          normalized *= Cast<float>(weight_data[column]);
        }
        input_row[column] = Cast<T>(normalized);
      }
    }
  }
};

}  // namespace infini::ops

#endif
