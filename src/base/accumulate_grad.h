#ifndef INFINI_OPS_BASE_ACCUMULATE_GRAD_H_
#define INFINI_OPS_BASE_ACCUMULATE_GRAD_H_

#include <cassert>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class AccumulateGrad : public Operator<AccumulateGrad> {
 public:
  AccumulateGrad(const Tensor gradient, const double rate, Tensor tensor)
      : gradient_shape_{gradient.shape()},
        gradient_strides_{gradient.strides()},
        gradient_type_{gradient.dtype()},
        tensor_shape_{tensor.shape()},
        tensor_strides_{tensor.strides()},
        tensor_type_{tensor.dtype()},
        rate_{rate},
        device_index_{tensor.device().index()} {
    AssertTrainingDevices({gradient, tensor});
    assert(gradient.shape() == tensor.shape() &&
           "`AccumulateGrad` requires `gradient` and `tensor` to share shape");
    assert(gradient_type_ == tensor_type_ &&
           "`AccumulateGrad` requires `gradient` and `tensor` to share dtype");
    assert(
        gradient.device() == tensor.device() &&
        "`AccumulateGrad` requires `gradient` and `tensor` on the same device");
  }

  virtual void operator()(const Tensor gradient, const double rate,
                          Tensor tensor) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  Tensor::Shape gradient_shape_;
  Tensor::Strides gradient_strides_;
  DataType gradient_type_;
  Tensor::Shape tensor_shape_;
  Tensor::Strides tensor_strides_;
  DataType tensor_type_;
  double rate_{1.0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_ACCUMULATE_GRAD_H_
