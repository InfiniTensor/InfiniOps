#ifndef INFINI_OPS_CPU_SWIGLU_SWIGLU_H_
#define INFINI_OPS_CPU_SWIGLU_SWIGLU_H_

#include <cmath>

#include "base/swiglu.h"
#include "common/generic_utils.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kCpu> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu{input, gate, out} {}

  void operator()(void* stream, const Tensor input, const Tensor gate,
                  Tensor out) const override {
    DispatchFunc<AllFloatTypes>(
        out_type_,
        [&]<typename T>() { compute<T>(stream, input, gate, out); },
        "Operator<Swiglu, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void compute(void* stream, const Tensor input, const Tensor gate,
               Tensor out) const {
    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* gate_ptr = static_cast<const T*>(gate.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i
                       : utils::indexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx =
          get_idx(i, is_input_contiguous_, input_shape_.data(),
                  input_strides_.data());
      auto gate_idx =
          get_idx(i, is_gate_contiguous_, gate_shape_.data(),
                  gate_strides_.data());
      auto out_idx =
          get_idx(i, is_out_contiguous_, out_shape_.data(), out_strides_.data());

      // SwiGLU(x, gate) = Swish(x) * gate
      // where Swish(x) = x * sigmoid(x)
      const T x = input_ptr[input_idx];
      const T sigmoid_x = static_cast<T>(1.0 / (1.0 + std::exp(-static_cast<double>(x))));
      const T swish_x = x * sigmoid_x;
      out_ptr[out_idx] = swish_x * gate_ptr[gate_idx];
    }
  }
};

}  // namespace infini::ops

#endif
