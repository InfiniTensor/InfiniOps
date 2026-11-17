#ifndef INFINI_OPS_BASE_MUL_BACKWARD_H_
#define INFINI_OPS_BASE_MUL_BACKWARD_H_

#include <cassert>

#include "base/validation.h"
#include "operator.h"

namespace infini::ops {

class MulBackward : public Operator<MulBackward> {
 public:
  MulBackward(const Tensor grad_output, const Tensor input, const Tensor other,
              Tensor grad_input, Tensor grad_other)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        grad_other_shape_{grad_other.shape()},
        grad_other_strides_{grad_other.strides()},
        grad_other_type_{grad_other.dtype()},
        device_index_{grad_input.device().index()} {
    validation::AssertSameDevice(
        {grad_output, input, other, grad_input, grad_other});
    assert(grad_output_type_ == grad_input_type_ &&
           grad_input_type_ == grad_other_type_ &&
           "`MulBackward` requires matching grad dtypes");
    assert(
        input_type_ == grad_output_type_ && other_type_ == grad_output_type_ &&
        "`MulBackward` requires input/other dtype to match grad_output dtype");

    assert(grad_input.shape() == input.shape() &&
           grad_other.shape() == other.shape() &&
           "Multiplication gradients must match their input shapes.");

    validation::AssertBroadcastsTo(input, grad_output);

    validation::AssertBroadcastsTo(other, grad_output);
  }

  // Aliasing: inputs may overlap each other and either output.
  // The storage byte spans of grad_input and grad_other must not overlap,
  // including gaps in non-contiguous layouts.
  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const Tensor other, Tensor grad_input,
                          Tensor grad_other) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  Tensor::Shape grad_other_shape_;

  Tensor::Strides grad_other_strides_;

  DataType grad_other_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MUL_BACKWARD_H_
