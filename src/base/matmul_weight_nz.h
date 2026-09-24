#ifndef INFINI_OPS_BASE_MATMUL_WEIGHT_NZ_H_
#define INFINI_OPS_BASE_MATMUL_WEIGHT_NZ_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

class MatmulWeightNz : public Operator<MatmulWeightNz> {
 public:
  MatmulWeightNz(const Tensor input, const Tensor weight, Tensor out)
      : rows_{input.size(0)},
        channels_{input.size(1)},
        outputs_{weight.size(1)} {
    assert(input.ndim() == 2 && weight.ndim() == 2 && out.ndim() == 2 &&
           "operator `MatmulWeightNz` requires 2D tensors");
    assert(input.size(1) == weight.size(0) &&
           "operator `MatmulWeightNz` input channels must match weight rows");
    assert(out.size(0) == input.size(0) &&
           "operator `MatmulWeightNz` output rows must match input");
    assert(out.size(1) == weight.size(1) &&
           "operator `MatmulWeightNz` output columns must match weight");
    assert(input.dtype() == weight.dtype() && input.dtype() == out.dtype() &&
           "operator `MatmulWeightNz` requires matching dtypes");
    assert((input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16) &&
           "operator `MatmulWeightNz` supports fp16 or bf16 only");
    assert(weight.stride(0) == static_cast<Tensor::Stride>(weight.size(1)) &&
           weight.stride(1) == 1 &&
           "operator `MatmulWeightNz` weight must be contiguous");
    assert(input.stride(0) == static_cast<Tensor::Stride>(input.size(1)) &&
           input.stride(1) == 1 &&
           "operator `MatmulWeightNz` input must be contiguous");
    assert(out.stride(0) == static_cast<Tensor::Stride>(out.size(1)) &&
           out.stride(1) == 1 &&
           "operator `MatmulWeightNz` output must be contiguous");
    assert(weight.size(0) % 16 == 0 && weight.size(1) % 16 == 0 &&
           "operator `MatmulWeightNz` requires 16-aligned weight shape");
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          Tensor out) const = 0;

 protected:
  Tensor::Size rows_;
  Tensor::Size channels_;
  Tensor::Size outputs_;
};

}  // namespace infini::ops

#endif
