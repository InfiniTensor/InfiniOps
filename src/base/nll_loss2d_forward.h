#ifndef INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_H_
#define INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_H_

#include "operator.h"

namespace infini::ops {

class NllLoss2dForward : public Operator<NllLoss2dForward> {
 public:
  NllLoss2dForward(const Tensor input, const Tensor target,
                   const int64_t reduction, const int64_t ignore_index,
                   Tensor output, Tensor total_weight)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        total_weight_shape_{total_weight.shape()},
        total_weight_strides_{total_weight.strides()},
        total_weight_type_{total_weight.dtype()},
        reduction_{reduction},
        ignore_index_{ignore_index},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor target,
                          const int64_t reduction, const int64_t ignore_index,
                          Tensor output, Tensor total_weight) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape total_weight_shape_;

  Tensor::Strides total_weight_strides_;

  DataType total_weight_type_;

  int64_t reduction_{};

  int64_t ignore_index_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
