#ifndef INFINI_OPS_CUDA_SWIGLU_DSL_H_
#define INFINI_OPS_CUDA_SWIGLU_DSL_H_

#include "cuda/templates/binary_elementwise.cuh"
#include "base/swiglu.h"

namespace infini::ops {

// Device-side binary functor for `Swiglu` (DSL).
template <Device::Type kDev>
struct DslSwigluOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    using ComputeType = float;
    auto va = Caster<kDev>::template Cast<ComputeType>(a);
    auto vb = Caster<kDev>::template Cast<ComputeType>(b);
    auto t2 = vb / (static_cast<ComputeType>(1) + expf(-vb));
    return Caster<kDev>::template Cast<T>((va * t2));
  }
};

template <typename Backend>
class DslCudaSwiglu : public Swiglu {
 public:
  DslCudaSwiglu(const Tensor input, const Tensor other, Tensor out)
      : Swiglu{input, other, out},
        brick_{input, other, out, ndim_} {}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    brick_.template Run<AllTypes, DslSwigluOp>(
        stream_, input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        out_type_);
  }

 private:
  BinaryElementwiseBrick<Backend> brick_;
};

}  // namespace infini::ops

#endif
