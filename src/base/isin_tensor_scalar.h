#ifndef INFINI_OPS_BASE_ISIN_TENSOR_SCALAR_H_
#define INFINI_OPS_BASE_ISIN_TENSOR_SCALAR_H_

#include "operator.h"

namespace infini::ops {

class IsinTensorScalar : public Operator<IsinTensorScalar> {
 public:
  IsinTensorScalar(const Tensor elements, const double test_element, Tensor out)
      : elements_shape_{elements.shape()},
        elements_strides_{elements.strides()},
        elements_type_{elements.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor elements, const double test_element,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape elements_shape_;

  Tensor::Strides elements_strides_;

  DataType elements_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
