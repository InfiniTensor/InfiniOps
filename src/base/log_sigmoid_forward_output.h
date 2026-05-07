#ifndef INFINI_OPS_BASE_LOG_SIGMOID_FORWARD_OUTPUT_H_
#define INFINI_OPS_BASE_LOG_SIGMOID_FORWARD_OUTPUT_H_

#include "operator.h"

namespace infini::ops {

class LogSigmoidForwardOutput : public Operator<LogSigmoidForwardOutput> {
 public:
  LogSigmoidForwardOutput(const Tensor self, Tensor output, Tensor buffer)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        buffer_shape_{buffer.shape()},
        buffer_strides_{buffer.strides()},
        buffer_type_{buffer.dtype()},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor self, Tensor output,
                          Tensor buffer) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape buffer_shape_;

  Tensor::Strides buffer_strides_;

  DataType buffer_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
