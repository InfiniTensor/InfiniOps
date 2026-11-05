#ifndef INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_
#define INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class MaskedFillBackward : public Operator<MaskedFillBackward> {
 public:
  MaskedFillBackward(const Tensor grad_output, const Tensor mask,
                     Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        mask_shape_{mask.shape()},
        mask_strides_{mask.strides()},
        mask_type_{mask.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {
    AssertTrainingDevices({grad_output, mask, grad_input});
    assert(
        grad_output.shape() == grad_input.shape() &&
        "`MaskedFillBackward` requires grad_output/grad_input to share shape");
    assert(
        grad_output_type_ == grad_input_type_ &&
        "`MaskedFillBackward` requires grad_output/grad_input dtype to match");
    assert(mask_type_ == DataType::kBool &&
           "`MaskedFillBackward` requires a bool `mask`");
    assert(mask.ndim() <= grad_output.ndim() &&
           "`MaskedFillBackward` mask rank must not exceed tensor rank");

    tail_aligned_ = AlignsAt(mask, grad_output, /*tail=*/true);
    if (!tail_aligned_) {
      assert(
          AlignsAt(mask, grad_output, /*tail=*/false) &&
          "`MaskedFillBackward` mask must align to leading or trailing dims");
    }
  }

  virtual void operator()(const Tensor grad_output, const Tensor mask,
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

  static bool AlignsAt(const Tensor mask, const Tensor tensor,
                       const bool tail) {
    const auto offset = tail ? tensor.ndim() - mask.ndim() : 0;
    for (Tensor::Size i = 0; i < mask.ndim(); ++i) {
      const auto dim = tensor.size(i + offset);
      if (mask.size(i) != 1 && mask.size(i) != dim) {
        return false;
      }
    }
    return true;
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape mask_shape_;
  Tensor::Strides mask_strides_;
  DataType mask_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  bool tail_aligned_{true};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_
