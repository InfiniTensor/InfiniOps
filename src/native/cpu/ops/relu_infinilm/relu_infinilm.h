#ifndef INFINI_OPS_CPU_RELU_INFINILM_RELU_INFINILM_H_
#define INFINI_OPS_CPU_RELU_INFINILM_RELU_INFINILM_H_

#include <type_traits>

#include "base/relu_infinilm.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kCpu> : public ReluInfinilm,
                                                   Caster<Device::Type::kCpu> {
 public:
  using ReluInfinilm::ReluInfinilm;

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "Operator<ReluInfinilm, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contiguous, const auto* shape,
                       const auto* strides) {
      return is_contiguous ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      const auto input_idx = get_idx(
          i, is_input_contiguous_, input_shape_.data(), input_strides_.data());
      const auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                                   out_strides_.data());
      const T value = input_ptr[input_idx];
      const ComputeType comparable = Cast<ComputeType>(value);
      out_ptr[out_idx] =
          comparable < ComputeType{0} ? Cast<T>(ComputeType{0}) : value;
    }
  }
};

}  // namespace infini::ops

#endif
