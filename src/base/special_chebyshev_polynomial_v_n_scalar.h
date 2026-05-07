#ifndef INFINI_OPS_BASE_SPECIAL_CHEBYSHEV_POLYNOMIAL_V_N_SCALAR_H_
#define INFINI_OPS_BASE_SPECIAL_CHEBYSHEV_POLYNOMIAL_V_N_SCALAR_H_

#include "operator.h"

namespace infini::ops {

class SpecialChebyshevPolynomialVNScalar
    : public Operator<SpecialChebyshevPolynomialVNScalar> {
 public:
  SpecialChebyshevPolynomialVNScalar(const Tensor x, const double n, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        x_type_{x.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor x, const double n, Tensor out) const = 0;

 protected:
  Tensor::Shape x_shape_;

  Tensor::Strides x_strides_;

  DataType x_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
