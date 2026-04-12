#ifndef INFINI_OPS_CUDA_MUL_DSL_H_
#define INFINI_OPS_CUDA_MUL_DSL_H_

#include "cuda/templates/binary_elementwise.cuh"
#include "base/mul.h"

namespace infini::ops {

// Device-side binary functor for `Mul` (DSL).
template <Device::Type kDev>
struct DslMulOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    using ComputeType = float;
    auto va = Caster<kDev>::template Cast<ComputeType>(a);
    auto vb = Caster<kDev>::template Cast<ComputeType>(b);
    return Caster<kDev>::template Cast<T>((va * vb));
  }
};

template <typename Backend>
class DslCudaMul : public Mul {
 public:
  DslCudaMul(const Tensor input, const Tensor other, Tensor out)
      : Mul{input, other, out},
        brick_{input, other, out, ndim_} {}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    brick_.template Run<AllTypes, DslMulOp>(
        stream_, input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        out_type_);
  }

 private:
  BinaryElementwiseBrick<Backend> brick_;
};

}  // namespace infini::ops

#endif
