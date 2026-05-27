#ifndef INFINI_OPS_BASE_POW_INPLACE_H_
#define INFINI_OPS_BASE_POW_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class PowInplace : public Operator<PowInplace> {
 public:
  PowInplace(Tensor input, const double exponent)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        exponent_{exponent},
        device_index_{input.device().index()} {}

  PowInplace(Tensor input, const Tensor exponent)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const double exponent) const = 0;

  virtual void operator()(Tensor input, const Tensor exponent) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  double exponent_{};

  Tensor::Shape exponent_shape_;

  Tensor::Strides exponent_strides_;

  DataType exponent_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
