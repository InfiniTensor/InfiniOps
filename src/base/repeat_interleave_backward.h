#ifndef INFINI_OPS_BASE_REPEAT_INTERLEAVE_BACKWARD_H_
#define INFINI_OPS_BASE_REPEAT_INTERLEAVE_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class RepeatInterleaveBackward : public Operator<RepeatInterleaveBackward> {
 public:
  RepeatInterleaveBackward(const Tensor grad_output, const int64_t dim,
                           Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        dim_{NormalizeDim(dim, grad_input.ndim())},
        device_index_{grad_input.device().index()} {
    assert(grad_output_type_ == grad_input_type_ &&
           "`RepeatInterleaveBackward` requires same dtype");
    assert(grad_output.ndim() == grad_input.ndim() &&
           "`RepeatInterleaveBackward` requires same rank");
    for (Tensor::Size i = 0; i < grad_input.ndim(); ++i) {
      if (i == dim_) continue;
      assert(grad_output.size(i) == grad_input.size(i) &&
             "`RepeatInterleaveBackward` shape mismatch on non-repeated dims");
    }
    assert(grad_input.size(dim_) > 0 &&
           "`RepeatInterleaveBackward` requires non-empty repeated dim");
    repeats_ = grad_output.size(dim_) / grad_input.size(dim_);
    assert(grad_output.size(dim_) == grad_input.size(dim_) * repeats_ &&
           "`RepeatInterleaveBackward` grad_output dim must be divisible");
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim,
                          Tensor grad_input) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`RepeatInterleaveBackward` requires ndim > 0");
    if (ndim == 0) return 0;
    const auto d = dim % static_cast<int64_t>(ndim);
    return d < 0 ? d + static_cast<int64_t>(ndim) : d;
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  int64_t repeats_{0};
  int64_t dim_{0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_REPEAT_INTERLEAVE_BACKWARD_H_
