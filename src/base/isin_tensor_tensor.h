#ifndef INFINI_OPS_BASE_ISIN_TENSOR_TENSOR_H_
#define INFINI_OPS_BASE_ISIN_TENSOR_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class IsinTensorTensor : public Operator<IsinTensorTensor> {
 public:
  IsinTensorTensor(const Tensor elements, const Tensor test_elements,
                   Tensor out)
      : elements_shape_{elements.shape()},
        elements_strides_{elements.strides()},
        elements_type_{elements.dtype()},
        test_elements_shape_{test_elements.shape()},
        test_elements_strides_{test_elements.strides()},
        test_elements_type_{test_elements.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor elements, const Tensor test_elements,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape elements_shape_;
  Tensor::Strides elements_strides_;
  DataType elements_type_;
  Tensor::Shape test_elements_shape_;
  Tensor::Strides test_elements_strides_;
  DataType test_elements_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
