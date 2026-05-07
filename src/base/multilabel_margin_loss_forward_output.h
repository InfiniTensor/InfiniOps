#ifndef INFINI_OPS_BASE_MULTILABEL_MARGIN_LOSS_FORWARD_OUTPUT_H_
#define INFINI_OPS_BASE_MULTILABEL_MARGIN_LOSS_FORWARD_OUTPUT_H_

#include "operator.h"

namespace infini::ops {

class MultilabelMarginLossForwardOutput
    : public Operator<MultilabelMarginLossForwardOutput> {
 public:
  MultilabelMarginLossForwardOutput(const Tensor self, const Tensor target,
                                    const int64_t reduction, Tensor output,
                                    Tensor is_target)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        target_shape_{target.shape()},
        target_strides_{target.strides()},
        target_type_{target.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        is_target_shape_{is_target.shape()},
        is_target_strides_{is_target.strides()},
        is_target_type_{is_target.dtype()},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor target,
                          const int64_t reduction, Tensor output,
                          Tensor is_target) const = 0;

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
  Tensor::Shape is_target_shape_;
  Tensor::Strides is_target_strides_;
  DataType is_target_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
