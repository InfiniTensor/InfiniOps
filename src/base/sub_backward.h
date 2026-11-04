#ifndef INFINI_OPS_BASE_SUB_BACKWARD_H_
#define INFINI_OPS_BASE_SUB_BACKWARD_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

class SubBackward : public Operator<SubBackward> {
 public:
  SubBackward(const Tensor grad_output, Tensor grad_a, Tensor grad_b)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_a_shape_{grad_a.shape()},
        grad_a_strides_{grad_a.strides()},
        grad_a_type_{grad_a.dtype()},
        grad_b_shape_{grad_b.shape()},
        grad_b_strides_{grad_b.strides()},
        grad_b_type_{grad_b.dtype()},
        device_index_{grad_a.device().index()} {
    assert(grad_output_type_ == grad_a_type_ && grad_a_type_ == grad_b_type_ &&
           "`SubBackward` requires matching dtypes");
    assert(grad_output.device() == grad_a.device() &&
           grad_a.device() == grad_b.device() &&
           "`SubBackward` requires grad_output/grad_a/grad_b on the same device");
    ValidateBroadcast(grad_a, grad_output);
    ValidateBroadcast(grad_b, grad_output);
  }

  virtual void operator()(const Tensor grad_output, Tensor grad_a,
                          Tensor grad_b) const = 0;

 protected:
  static void ValidateBroadcast(const Tensor input, const Tensor out) {
    assert(input.ndim() <= out.ndim() &&
           "`SubBackward` input rank must not exceed grad_output rank");
    const auto offset = out.ndim() - input.ndim();
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      assert((input.size(i) == 1 || input.size(i) == out.size(i + offset)) &&
             "`SubBackward` input must broadcast (right-aligned) to grad_output");
    }
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape grad_a_shape_;
  Tensor::Strides grad_a_strides_;
  DataType grad_a_type_;
  Tensor::Shape grad_b_shape_;
  Tensor::Strides grad_b_strides_;
  DataType grad_b_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_SUB_BACKWARD_H_