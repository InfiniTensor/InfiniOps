#ifndef INFINI_OPS_BASE_MERGED_QK_RMS_NORM_H_
#define INFINI_OPS_BASE_MERGED_QK_RMS_NORM_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class MergedQkRmsNorm : public Operator<MergedQkRmsNorm> {
 public:
  MergedQkRmsNorm(const Tensor input, const Tensor weight, float epsilon,
                  Tensor output)
      : input_shape_{input.shape()},
        output_shape_{output.shape()},
        weight_shape_{weight.shape()},
        epsilon_{epsilon},
        num_heads_{input.size(1)},
        normalized_width_{input.size(2)} {
    assert(input.ndim() == 3 && output.ndim() == 3 &&
           "`MergedQkRmsNorm` requires rank-3 packed Q/K tensors");
    assert(input.shape() == output.shape() &&
           "`MergedQkRmsNorm` requires matching input and output shapes");
    assert(input.stride(2) == 1 && output.stride(2) == 1 &&
           "`MergedQkRmsNorm` requires contiguous head dimensions");
    assert(weight.ndim() == 2 && weight.size(0) == num_heads_ &&
           weight.size(1) == normalized_width_ &&
           "`MergedQkRmsNorm` requires one weight row per packed head");
    assert(weight.stride(1) == 1 &&
           "`MergedQkRmsNorm` requires contiguous weight rows");
    assert(input.dtype() == output.dtype() &&
           output.dtype() == weight.dtype() &&
           "`MergedQkRmsNorm` requires matching tensor dtypes");
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          float epsilon, Tensor output) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Shape output_shape_;

  Tensor::Shape weight_shape_;

  float epsilon_{};

  Tensor::Size num_heads_{0};

  Tensor::Size normalized_width_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MERGED_QK_RMS_NORM_H_
