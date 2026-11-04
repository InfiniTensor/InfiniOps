#ifndef INFINI_OPS_BASE_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_
#define INFINI_OPS_BASE_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class VocabParallelCrossEntropyBackward
    : public Operator<VocabParallelCrossEntropyBackward> {
 public:
  VocabParallelCrossEntropyBackward(
      const Tensor grad_output, const Tensor softmax_local,
      const Tensor target_mask, const Tensor masked_target,
      const Tensor valid_mask_local, const int64_t vocab_size_local,
      const int64_t vocab_size_original, const double label_smoothing,
      Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        softmax_shape_{softmax_local.shape()},
        softmax_strides_{softmax_local.strides()},
        softmax_type_{softmax_local.dtype()},
        target_mask_shape_{target_mask.shape()},
        target_mask_strides_{target_mask.strides()},
        target_mask_type_{target_mask.dtype()},
        masked_target_shape_{masked_target.shape()},
        masked_target_strides_{masked_target.strides()},
        masked_target_type_{masked_target.dtype()},
        valid_mask_shape_{valid_mask_local.shape()},
        valid_mask_strides_{valid_mask_local.strides()},
        valid_mask_type_{valid_mask_local.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        vocab_size_local_{vocab_size_local},
        vocab_size_original_{vocab_size_original},
        label_smoothing_{label_smoothing},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(
      const Tensor grad_output, const Tensor softmax_local,
      const Tensor target_mask, const Tensor masked_target,
      const Tensor valid_mask_local, const int64_t vocab_size_local,
      const int64_t vocab_size_original, const double label_smoothing,
      Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;

  Tensor::Shape softmax_shape_;
  Tensor::Strides softmax_strides_;
  DataType softmax_type_;

  Tensor::Shape target_mask_shape_;
  Tensor::Strides target_mask_strides_;
  DataType target_mask_type_;

  Tensor::Shape masked_target_shape_;
  Tensor::Strides masked_target_strides_;
  DataType masked_target_type_;

  Tensor::Shape valid_mask_shape_;
  Tensor::Strides valid_mask_strides_;
  DataType valid_mask_type_;

  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;

  int64_t vocab_size_local_{0};
  int64_t vocab_size_original_{0};
  double label_smoothing_{0.0};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_
