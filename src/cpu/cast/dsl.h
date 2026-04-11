#ifndef INFINI_OPS_CPU_CAST_DSL_H_
#define INFINI_OPS_CPU_CAST_DSL_H_

#include "cpu/templates/unary_elementwise.h"
#include "base/cast.h"
#include "impl.h"
#include "cpu/cast/registry.h"

namespace infini::ops {

// Host-side unary functor for `Cast` (CPU, DSL).
struct DslCpuCastOp {
  template <typename TIn, typename TOut>
  TOut operator()(const TIn& x) const {
    auto va = Caster<Device::Type::kCpu>::Cast<float>(x);
    return Caster<Device::Type::kCpu>::Cast<TOut>(va);
  }
};

template <>
class Operator<Cast, Device::Type::kCpu, Impl::kDsl> : public Cast {
 public:
  using Cast::Cast;

  void operator()(const Tensor input, Tensor out) const override {
    CpuUnaryElementwise<AllTypes, AllTypes>(
        input, out, output_size_, ndim_,
        is_input_contiguous_, is_out_contiguous_,
        input_shape_, out_shape_,
        input_strides_, out_strides_,
        input_dtype_, out_dtype_, DslCpuCastOp{});
  }
};

}  // namespace infini::ops

#endif
