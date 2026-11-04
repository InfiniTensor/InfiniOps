#ifndef INFINI_OPS_BASE_CROSS_ENTROPY_BACKWARD_H_
#define INFINI_OPS_BASE_CROSS_ENTROPY_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

class CrossEntropyBackward : public Operator<CrossEntropyBackward> {
 public:
  // Keep this synchronized with CrossEntropy's forward contract.
  static constexpr int64_t kIgnoreIndex = -100;

  CrossEntropyBackward(const Tensor input, const Tensor target,
                       const Tensor grad_output, Tensor grad_input)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {
    assert(input.ndim() >= 2 &&
           "`CrossEntropyBackward` requires input rank >= 2");
    assert(input.device() == target.device() &&
           target.device() == grad_output.device() &&
           grad_output.device() == grad_input.device() &&
           "`CrossEntropyBackward` requires all tensors on the same device");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32 ||
            input_type_ == DataType::kFloat64) &&
           "`CrossEntropyBackward` requires floating-point logits");
    assert((target_type_ == DataType::kInt64 ||
            target_type_ == DataType::kInt32 ||
            target_type_ == DataType::kUInt8) &&
           "`CrossEntropyBackward` supports int64/int32/uint8 targets");
    assert(target.ndim() == input.ndim() - 1 &&
           "`CrossEntropyBackward` target rank must be input rank - 1");
    for (Tensor::Size i = 0; i < target.ndim(); ++i) {
      const auto input_dim = i == 0 ? 0 : i + 1;
      assert(target.size(i) == input.size(input_dim) &&
             "`CrossEntropyBackward` target shape mismatch");
    }
    assert(grad_input.shape() == input.shape() &&
           "`CrossEntropyBackward` grad_input shape must match input");
    assert(grad_output.ndim() == 0 &&
           "`CrossEntropyBackward` requires scalar grad_output");
    assert(grad_output_type_ == input_type_ &&
           grad_input_type_ == input_type_ &&
           "`CrossEntropyBackward` requires matching floating-point dtypes");
  }

  virtual void operator()(const Tensor input, const Tensor target,
                          const Tensor grad_output,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape input_shape_;
  Tensor::Strides input_strides_;
  DataType input_type_;

  Tensor::Shape target_shape_;
  Tensor::Strides target_strides_;
  DataType target_type_;

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;

  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_CROSS_ENTROPY_BACKWARD_H_
