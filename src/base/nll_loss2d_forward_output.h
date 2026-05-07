#ifndef INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_OUTPUT_H_
#define INFINI_OPS_BASE_NLL_LOSS2D_FORWARD_OUTPUT_H_

#include "operator.h"

namespace infini::ops {

class NllLoss2dForwardOutput : public Operator<NllLoss2dForwardOutput> {
 public:
  NllLoss2dForwardOutput(const Tensor self, const Tensor target,
                         const int64_t reduction, const int64_t ignore_index,
                         Tensor output, Tensor total_weight)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        total_weight_shape_{total_weight.shape()},
        total_weight_strides_{total_weight.strides()},
        total_weight_type_{total_weight.dtype()},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor target,
                          const int64_t reduction, const int64_t ignore_index,
                          Tensor output, Tensor total_weight) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape target_shape_;

  Tensor::Strides target_strides_;

  DataType target_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape total_weight_shape_;

  Tensor::Strides total_weight_strides_;

  DataType total_weight_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
