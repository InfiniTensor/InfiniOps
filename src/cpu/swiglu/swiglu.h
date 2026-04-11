#ifndef INFINI_OPS_CPU_SWIGLU_SWIGLU_H_
#define INFINI_OPS_CPU_SWIGLU_SWIGLU_H_

#include <cmath>

#include "base/swiglu.h"
#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "cpu/swiglu/registry.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kCpu> : public Swiglu,
                                             Caster<Device::Type::kCpu> {
 public:
  using Swiglu::Swiglu;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, other, out);
        },
        "Operator<Swiglu, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor other, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* other_ptr = static_cast<const T*>(other.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                               input_strides_.data());
      auto gate_idx = get_idx(i, is_other_contiguous_, other_shape_.data(),
                              other_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());
      const ComputeType other_val = Cast<ComputeType>(other_ptr[gate_idx]);
      const ComputeType sigmoid_other = static_cast<ComputeType>(
          1.0 / (1.0 + std::exp(-static_cast<double>(other_val))));
      const ComputeType swish_other = other_val * sigmoid_other;
      out_ptr[out_idx] =
          Cast<T>(Cast<ComputeType>(input_ptr[input_idx]) * swish_other);
    }
  }
};

}  // namespace infini::ops

#endif
