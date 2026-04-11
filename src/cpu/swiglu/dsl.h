#ifndef INFINI_OPS_CPU_SWIGLU_DSL_H_
#define INFINI_OPS_CPU_SWIGLU_DSL_H_

#include "cpu/templates/binary_elementwise.h"
#include "base/swiglu.h"
#include "impl.h"
#include "cpu/swiglu/registry.h"

namespace infini::ops {

// Host-side binary functor for `Swiglu` (CPU, DSL).
struct DslCpuSwigluOp {
  template <typename T>
  T operator()(const T& a, const T& b) const {
    using ComputeType = float;
    auto va = static_cast<ComputeType>(a);
    auto vb = static_cast<ComputeType>(b);
    auto t2 = vb / (static_cast<ComputeType>(1) + std::exp(-vb));
    return static_cast<T>((va * t2));
  }
};

template <>
class Operator<Swiglu, Device::Type::kCpu, Impl::kDsl> : public Swiglu {
 public:
  using Swiglu::Swiglu;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    CpuBinaryElementwise<AllTypes>(
        input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        input_shape_, other_shape_, out_shape_,
        input_strides_, other_strides_, out_strides_,
        out_type_, DslCpuSwigluOp{});
  }
};

}  // namespace infini::ops

#endif
