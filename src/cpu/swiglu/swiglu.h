#ifndef INFINI_OPS_CPU_SWIGLU_SWIGLU_H_
#define INFINI_OPS_CPU_SWIGLU_SWIGLU_H_

#include <cmath>

#include "base/swiglu.h"
#include "common/generic_utils.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kCpu> : public Swiglu {
 public:
  using Swiglu::Swiglu;

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    DispatchFunc<AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, gate, out);
        },
        "Operator<Swiglu, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor gate, Tensor out) const {
    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* gate_ptr = static_cast<const T*>(gate.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                               input_strides_.data());
      auto gate_idx = get_idx(i, is_gate_contiguous_, gate_shape_.data(),
                              gate_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());

      const auto x{static_cast<double>(input_ptr[input_idx])};
      const auto gate_val{static_cast<double>(gate_ptr[gate_idx])};

      const auto sigmoid_x{1.0 / (1.0 + std::exp(-x))};
      const auto swish_x{x * sigmoid_x};

      out_ptr[out_idx] = static_cast<T>(swish_x * gate_val);
    }
  }
};

}  // namespace infini::ops

#endif
