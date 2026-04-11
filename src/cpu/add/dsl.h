#ifndef INFINI_OPS_CPU_ADD_DSL_H_
#define INFINI_OPS_CPU_ADD_DSL_H_

#include "cpu/templates/binary_elementwise.h"
#include "base/add.h"
#include "impl.h"
#include "cpu/add/registry.h"

namespace infini::ops {

// Host-side binary functor for `Add` (CPU, DSL).
struct DslCpuAddOp {
  template <typename T>
  T operator()(const T& a, const T& b) const {
    using ComputeType = float;
    auto va = static_cast<ComputeType>(a);
    auto vb = static_cast<ComputeType>(b);
    return static_cast<T>((va + vb));
  }
};

template <>
class Operator<Add, Device::Type::kCpu, Impl::kDsl> : public Add {
 public:
  using Add::Add;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    CpuBinaryElementwise<AllTypes>(
        input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        input_shape_, other_shape_, out_shape_,
        input_strides_, other_strides_, out_strides_,
        out_type_, DslCpuAddOp{});
  }
};

}  // namespace infini::ops

#endif
