#ifndef INFINI_OPS_BASE_HARDTANH_BACKWARD_GRAD_INPUT_H_
#define INFINI_OPS_BASE_HARDTANH_BACKWARD_GRAD_INPUT_H_

#include "operator.h"

namespace infini::ops {

class HardtanhBackwardGradInput : public Operator<HardtanhBackwardGradInput> {
 public:
  HardtanhBackwardGradInput(const Tensor grad_output, const Tensor self,
                            const double min_val, const double max_val,
                            Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor self,
                          const double min_val, const double max_val,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
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
