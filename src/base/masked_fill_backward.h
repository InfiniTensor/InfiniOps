#ifndef INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_
#define INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "base/validation.h"
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
    validation::AssertSameDevice({grad_output, mask, grad_input});
    assert(
        grad_output_type_ == grad_input_type_ &&
        "`MaskedFillBackward` requires grad_output/grad_input dtype to match");
    assert(mask_type_ == DataType::kBool &&
           "`MaskedFillBackward` requires a bool `mask`");
    assert(IsValidBroadcastShape(grad_input_shape_, mask_shape_,
                                 grad_output_shape_) &&
           "`MaskedFillBackward` requires grad_output shape to equal the "
           "broadcast shape of grad_input and mask");
  }

  virtual void operator()(const Tensor grad_output, const Tensor mask,
                          Tensor grad_input) const = 0;

 protected:
  static bool IsValidBroadcastShape(const Tensor::Shape& input_shape,
                                    const Tensor::Shape& mask_shape,
                                    const Tensor::Shape& output_shape) {
    const auto ndim = input_shape.size() > mask_shape.size()
                          ? input_shape.size()
                          : mask_shape.size();
    if (output_shape.size() != ndim) {
      return false;
    }

    const auto input_offset = ndim - input_shape.size();

    const auto mask_offset = ndim - mask_shape.size();

    for (std::size_t i = 0; i < ndim; ++i) {
      const auto input_dim =
          i < input_offset ? 1 : input_shape[i - input_offset];
      const auto mask_dim = i < mask_offset ? 1 : mask_shape[i - mask_offset];

      if (input_dim != mask_dim && input_dim != 1 && mask_dim != 1) {
        return false;
      }

      const auto output_dim = input_dim == 1 ? mask_dim : input_dim;
      if (output_shape[i] != output_dim) {
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

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MASKED_FILL_BACKWARD_H_
