#ifndef INFINI_OPS_BASE_FILL_H_
#define INFINI_OPS_BASE_FILL_H_

#include "operator.h"

namespace infini::ops {

class Fill : public Operator<Fill> {
 public:
  Fill(Tensor input, const double value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        value_{value},
        device_index_{input.device().index()} {}

  Fill(Tensor input, const Tensor value)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        value_shape_{value.shape()},
        value_strides_{value.strides()},
        value_type_{value.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const double value) const = 0;

  virtual void operator()(Tensor input, const Tensor value) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  double value_{};

  Tensor::Shape value_shape_;

  Tensor::Strides value_strides_;

  DataType value_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
