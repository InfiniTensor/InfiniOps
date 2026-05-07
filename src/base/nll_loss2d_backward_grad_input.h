#ifndef INFINI_OPS_BASE_NLL_LOSS2D_BACKWARD_GRAD_INPUT_H_
#define INFINI_OPS_BASE_NLL_LOSS2D_BACKWARD_GRAD_INPUT_H_

#include "operator.h"

namespace infini::ops {

class NllLoss2dBackwardGradInput : public Operator<NllLoss2dBackwardGradInput> {
 public:
  NllLoss2dBackwardGradInput(const Tensor grad_output, const Tensor self,
                             const Tensor target, const int64_t reduction,
                             const int64_t ignore_index,
                             const Tensor total_weight, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        total_weight_shape_{total_weight.shape()},
        total_weight_strides_{total_weight.strides()},
        total_weight_type_{total_weight.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor self,
                          const Tensor target, const int64_t reduction,
                          const int64_t ignore_index, const Tensor total_weight,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape total_weight_shape_;

  Tensor::Strides total_weight_strides_;

  DataType total_weight_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
