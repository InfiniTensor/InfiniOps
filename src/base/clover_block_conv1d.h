#pragma once

#include "operator.h"

namespace infini::ops {

// Clover preview convolution reads a clean prefix without updating its state.
class CloverBlockConv1d : public Operator<CloverBlockConv1d> {
 public:
  CloverBlockConv1d(const Tensor input, const Tensor state,
                    const Tensor indices, const Tensor weight,
                    const Tensor right_weight, const Tensor valid,
                    size_t block_size, Tensor out) {
    assert((input.ndim() == 2 && out.shape() == input.shape() &&
            state.ndim() == 3 && weight.ndim() == 3 &&
            weight.shape() == Tensor::Shape{input.size(1), 1, 4} &&
            right_weight.shape() == Tensor::Shape{input.size(1), 1, 3} &&
            state.size(1) == input.size(1) && state.size(2) == 3 &&
            indices.ndim() == 1 && block_size > 0 && block_size <= 32 &&
            input.size(0) == indices.numel() * block_size &&
            valid.shape() == Tensor::Shape{input.size(0)}) &&
           "CloverBlockConv1d: invalid block or state shape.");
    assert((input.dtype() == DataType::kBFloat16 ||
            input.dtype() == DataType::kFloat16) &&
           "CloverBlockConv1d: expected BF16 or FP16 input.");
    for (const auto& tensor : {state, weight, right_weight, out}) {
      assert(tensor.dtype() == input.dtype() &&
             "CloverBlockConv1d: input and weight dtypes must match.");
    }
    assert((indices.dtype() == DataType::kInt32 ||
            indices.dtype() == DataType::kInt64) &&
           indices.IsContiguous() && valid.dtype() == DataType::kInt8 &&
           valid.IsContiguous() &&
           "CloverBlockConv1d: expected contiguous indices and int8 validity.");
    for (const auto& tensor :
         {state, indices, weight, right_weight, valid, out}) {
      assert(tensor.device() == input.device() &&
             "CloverBlockConv1d: device mismatch.");
    }
    assert(!out.HasBroadcastDim() && "CloverBlockConv1d: overlapping output.");
  }

  virtual void operator()(const Tensor input, const Tensor state,
                          const Tensor indices, const Tensor weight,
                          const Tensor right_weight, const Tensor valid,
                          size_t block_size, Tensor out) const = 0;
};

}  // namespace infini::ops
