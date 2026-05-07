#ifndef INFINI_OPS_BASE_HARDSHRINK_BACKWARD_GRAD_INPUT_H_
#define INFINI_OPS_BASE_HARDSHRINK_BACKWARD_GRAD_INPUT_H_

#include "operator.h"

namespace infini::ops {

class HardshrinkBackwardGradInput
    : public Operator<HardshrinkBackwardGradInput> {
 public:
  HardshrinkBackwardGradInput(const Tensor grad_out, const Tensor self,
                              const double lambd, Tensor grad_input)
      : grad_out_shape_{grad_out.shape()},
        grad_out_strides_{grad_out.strides()},
        grad_out_type_{grad_out.dtype()},
        self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_out, const Tensor self,
                          const double lambd, Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_out_shape_;
  Tensor::Strides grad_out_strides_;
  DataType grad_out_type_;
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape grad_input_shape_;
  Tensor::Strides grad_input_strides_;
  DataType grad_input_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
