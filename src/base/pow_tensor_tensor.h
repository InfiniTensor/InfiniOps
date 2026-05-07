#ifndef INFINI_OPS_BASE_POW_TENSOR_TENSOR_H_
#define INFINI_OPS_BASE_POW_TENSOR_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class PowTensorTensor : public Operator<PowTensorTensor> {
 public:
  PowTensorTensor(const Tensor self, const Tensor exponent, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor exponent,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape exponent_shape_;
  Tensor::Strides exponent_strides_;
  DataType exponent_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
