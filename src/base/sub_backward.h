#ifndef INFINI_OPS_BASE_SUB_BACKWARD_H_
#define INFINI_OPS_BASE_SUB_BACKWARD_H_

#include <cassert>

#include "base/validation.h"
#include "operator.h"

namespace infini::ops {

class SubBackward : public Operator<SubBackward> {
 public:
  SubBackward(const Tensor grad_output, const double alpha, Tensor grad_input,
              Tensor grad_other)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        grad_other_shape_{grad_other.shape()},
        grad_other_strides_{grad_other.strides()},
        grad_other_type_{grad_other.dtype()},
        device_index_{grad_input.device().index()} {
    validation::AssertSameDevice({grad_output, grad_input, grad_other});
    assert(grad_output_type_ == grad_input_type_ &&
           grad_input_type_ == grad_other_type_ &&
           "`SubBackward` requires matching dtypes");

    validation::AssertBroadcastsTo(grad_input, grad_output);

    validation::AssertBroadcastsTo(grad_other, grad_output);
  }

  // Inputs may alias outputs; output byte spans must not overlap.
  virtual void operator()(const Tensor grad_output, const double alpha,
                          Tensor grad_input, Tensor grad_other) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  Tensor::Shape grad_other_shape_;

  Tensor::Strides grad_other_strides_;

  DataType grad_other_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SUB_BACKWARD_H_
