#ifndef INFINI_OPS_BASE_TRANSPOSE_BACKWARD_H_
#define INFINI_OPS_BASE_TRANSPOSE_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class TransposeBackward : public Operator<TransposeBackward> {
 public:
  TransposeBackward(const Tensor grad_output, const int64_t dim0,
                    const int64_t dim1, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        dim0_{NormalizeDim(dim0, grad_output.ndim())},
        dim1_{NormalizeDim(dim1, grad_output.ndim())},
        device_index_{grad_input.device().index()} {
    assert(grad_output.ndim() >= 2 &&
           "`TransposeBackward` requires at least 2 dims");
    assert(dim0_ != dim1_ &&
           "`TransposeBackward` requires distinct dims");
    assert(grad_output_type_ == grad_input_type_ &&
           "`TransposeBackward` requires same dtype");

    for (Tensor::Size i = 0; i < grad_output.ndim(); ++i) {
      const auto expect = (i == dim0_) ? grad_output.size(dim1_)
                          : (i == dim1_) ? grad_output.size(dim0_)
                                         : grad_output.size(i);
      assert(grad_input.size(i) == expect &&
             "`TransposeBackward` grad_input shape mismatch");
    }
  }

  virtual void operator()(const Tensor grad_output, const int64_t dim0,
                          const int64_t dim1, Tensor grad_input) const = 0;

 protected:
  static int64_t NormalizeDim(const int64_t dim, const Tensor::Size ndim) {
    assert(ndim > 0 && "`TransposeBackward` requires ndim > 0");
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

  int64_t dim0_{0};
  int64_t dim1_{0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TRANSPOSE_BACKWARD_H_
