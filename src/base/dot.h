#ifndef INFINI_OPS_BASE_DOT_H_
#define INFINI_OPS_BASE_DOT_H_

#include "operator.h"

namespace infini::ops {

class Dot : public Operator<Dot> {
 public:
  Dot(const Tensor self, const Tensor tensor, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        tensor_shape_{tensor.shape()},
        tensor_strides_{tensor.strides()},
        tensor_type_{tensor.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor tensor,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape tensor_shape_;
  Tensor::Strides tensor_strides_;
  DataType tensor_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
