#ifndef INFINI_OPS_BASE_FREXP_TENSOR_H_
#define INFINI_OPS_BASE_FREXP_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class FrexpTensor : public Operator<FrexpTensor> {
 public:
  FrexpTensor(const Tensor self, Tensor mantissa, Tensor exponent)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        mantissa_shape_{mantissa.shape()},
        mantissa_strides_{mantissa.strides()},
        mantissa_type_{mantissa.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        device_index_{mantissa.device().index()} {}

  virtual void operator()(const Tensor self, Tensor mantissa,
                          Tensor exponent) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape mantissa_shape_;
  Tensor::Strides mantissa_strides_;
  DataType mantissa_type_;
  Tensor::Shape exponent_shape_;
  Tensor::Strides exponent_strides_;
  DataType exponent_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
