#ifndef INFINI_OPS_BASE_CROSS_ENTROPY_BACKWARD_H_
#define INFINI_OPS_BASE_CROSS_ENTROPY_BACKWARD_H_

#include <cassert>
#include <cstdint>

#include "base/cross_entropy_common.h"
#include "operator.h"

namespace infini::ops {

class CrossEntropyBackward : public Operator<CrossEntropyBackward> {
 public:
  static constexpr int64_t kIgnoreIndex = -100;

  CrossEntropyBackward(const Tensor input, const Tensor target,
                       const Tensor grad_output, std::optional<Tensor> weight,
                       int64_t ignore_index, std::string reduction,
                       double label_smoothing, Tensor grad_input)
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
    ValidateCrossEntropy(input, target, weight, ignore_index, reduction,
                         label_smoothing, grad_output);
    validation::AssertSameDevice({input, grad_input});
    assert(grad_input.shape() == input.shape() &&
           grad_input.dtype() == input.dtype() &&
           "`CrossEntropyBackward` grad_input must match logits metadata.");
  }

  virtual void operator()(const Tensor input, const Tensor target,
                          const Tensor grad_output,
                          std::optional<Tensor> weight, int64_t ignore_index,
                          std::string reduction, double label_smoothing,
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
