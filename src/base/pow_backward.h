#ifndef INFINI_OPS_BASE_POW_BACKWARD_H_
#define INFINI_OPS_BASE_POW_BACKWARD_H_

#include <cassert>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class PowBackward : public Operator<PowBackward> {
 public:
  PowBackward(const Tensor grad_output, const Tensor input, const double scalar,
              const bool scalar_is_base, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        scalar_{scalar},
        scalar_is_base_{scalar_is_base},
        device_index_{grad_input.device().index()} {
    AssertTrainingDevices({grad_output, input, grad_input});
    assert(
        input.shape() == grad_output.shape() &&
        grad_output.shape() == grad_input.shape() &&
        "`PowBackward` requires input/grad_output/grad_input to share shape");
    assert(input_type_ == grad_output_type_ &&
           grad_output_type_ == grad_input_type_ &&
           "`PowBackward` requires matching dtypes");
    assert((!scalar_is_base_ || scalar_ > 0.0) &&
           "`PowBackward` scalar base must be positive for log(scalar).");
    assert(IsTrainingFloat(input_type_) &&
           "Power gradients require floating dtypes.");
  }

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const double scalar, const bool scalar_is_base,
                          Tensor grad_input) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  static bool IsTrainingFloat(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32 || dtype == DataType::kFloat64;
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  double scalar_{0.0};
  bool scalar_is_base_{false};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_POW_BACKWARD_H_
