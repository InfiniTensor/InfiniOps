#ifndef INFINI_OPS_BASE_MUL_BACKWARD_H_
#define INFINI_OPS_BASE_MUL_BACKWARD_H_

#include <cassert>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class MulBackward : public Operator<MulBackward> {
 public:
  MulBackward(const Tensor grad_output, const Tensor a, const Tensor b,
              Tensor grad_a, Tensor grad_b)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        a_shape_{a.shape()},
        a_strides_{a.strides()},
        a_type_{a.dtype()},
        b_shape_{b.shape()},
        b_strides_{b.strides()},
        b_type_{b.dtype()},
        grad_a_shape_{grad_a.shape()},
        grad_a_strides_{grad_a.strides()},
        grad_a_type_{grad_a.dtype()},
        grad_b_shape_{grad_b.shape()},
        grad_b_strides_{grad_b.strides()},
        grad_b_type_{grad_b.dtype()},
        device_index_{grad_a.device().index()} {
    AssertTrainingDevices({grad_output, a, b, grad_a, grad_b});
    assert(grad_output_type_ == grad_a_type_ && grad_a_type_ == grad_b_type_ &&
           "`MulBackward` requires matching grad dtypes");
    assert(a_type_ == grad_output_type_ && b_type_ == grad_output_type_ &&
           "`MulBackward` requires a/b dtype to match grad_output dtype");
    assert(grad_output.device() == a.device() && a.device() == b.device() &&
           "`MulBackward` requires all tensors on the same device");
    assert(grad_a.shape() == a.shape() && grad_b.shape() == b.shape() &&
           "Multiplication gradients must match their input shapes.");
    ValidateBroadcast(a, grad_output);
    ValidateBroadcast(b, grad_output);
    ValidateBroadcast(grad_a, grad_output);
    ValidateBroadcast(grad_b, grad_output);
  }

  virtual void operator()(const Tensor grad_output, const Tensor a,
                          const Tensor b, Tensor grad_a,
                          Tensor grad_b) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static void ValidateBroadcast(const Tensor input, const Tensor out) {
    assert(input.ndim() <= out.ndim() &&
           "`MulBackward` input rank must not exceed grad_output rank");
    const auto offset = out.ndim() - input.ndim();
    for (Tensor::Size i = 0; i < input.ndim(); ++i) {
      assert(
          (input.size(i) == 1 || input.size(i) == out.size(i + offset)) &&
          "`MulBackward` input must broadcast (right-aligned) to grad_output");
    }
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape a_shape_;
  Tensor::Strides a_strides_;
  DataType a_type_;
  Tensor::Shape b_shape_;
  Tensor::Strides b_strides_;
  DataType b_type_;
  Tensor::Shape grad_a_shape_;
  Tensor::Strides grad_a_strides_;
  DataType grad_a_type_;
  Tensor::Shape grad_b_shape_;
  Tensor::Strides grad_b_strides_;
  DataType grad_b_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MUL_BACKWARD_H_
