#ifndef INFINI_OPS_CUDA_CAST_DSL_H_
#define INFINI_OPS_CUDA_CAST_DSL_H_

#include "cuda/templates/unary_elementwise.cuh"
#include "base/cast.h"

namespace infini::ops {

// Device-side unary functor for `Cast` (DSL).
template <Device::Type kDev>
struct DslCastOp {
  template <typename TIn, typename TOut>
  __device__ __forceinline__ TOut operator()(const TIn& x) const {
    auto va = Caster<kDev>::template Cast<float>(x);
    return Caster<kDev>::template Cast<TOut>(va);
  }
};

template <typename Backend>
class DslCudaCast : public Cast {
 public:
  DslCudaCast(const Tensor input, Tensor out)
      : Cast{input, out},
        brick_{input, out, ndim_} {}

  void operator()(const Tensor input, Tensor out) const override {
    brick_.template Run<AllTypes, AllTypes, DslCastOp>(
        stream_, input, out, output_size_, ndim_,
        is_input_contiguous_, is_out_contiguous_,
        input_dtype_, out_dtype_);
  }

 private:
  UnaryElementwiseBrick<Backend> brick_;
};

}  // namespace infini::ops

#endif
