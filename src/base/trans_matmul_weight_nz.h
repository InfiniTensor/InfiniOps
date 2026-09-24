#ifndef INFINI_OPS_BASE_TRANS_MATMUL_WEIGHT_NZ_H_
#define INFINI_OPS_BASE_TRANS_MATMUL_WEIGHT_NZ_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

class TransMatmulWeightNz : public Operator<TransMatmulWeightNz> {
 public:
  TransMatmulWeightNz(const Tensor input, Tensor output)
      : rows_{input.size(0)}, columns_{input.size(1)} {
    assert(input.ndim() == 2 && output.ndim() == 2 &&
           "operator `TransMatmulWeightNz` requires 2D tensors");
    assert(input.shape() == output.shape() &&
           "operator `TransMatmulWeightNz` input and output shapes differ");
    assert(input.dtype() == output.dtype() &&
           (input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16) &&
           "operator `TransMatmulWeightNz` supports fp16 or bf16 only");
    assert(input.stride(0) == static_cast<Tensor::Stride>(input.size(1)) &&
           input.stride(1) == 1 &&
           "operator `TransMatmulWeightNz` input must be contiguous");
    assert(output.stride(0) == static_cast<Tensor::Stride>(output.size(1)) &&
           output.stride(1) == 1 &&
           "operator `TransMatmulWeightNz` output must be contiguous");
    assert(rows_ % 16 == 0 && columns_ % 16 == 0 &&
           "operator `TransMatmulWeightNz` requires a 16-aligned shape");
  }

  virtual void operator()(const Tensor input, Tensor output) const = 0;

 protected:
  Tensor::Size rows_;
  Tensor::Size columns_;
};

}  // namespace infini::ops

#endif
