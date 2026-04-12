#ifndef INFINI_OPS_BASE_SWIGLU_H_
#define INFINI_OPS_BASE_SWIGLU_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Swiglu : public Operator<Swiglu> {
 public:
  Swiglu(const Tensor input, const Tensor other, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        other_type_{other.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        other_shape_{other.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        other_strides_{other.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_other_contiguous_{other.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(
        input_type_ == other_type_ && other_type_ == out_type_ &&
        "operator `Swiglu` requires all input and output tensors to have the "
        "same dtype");
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType other_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape other_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides other_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_other_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
