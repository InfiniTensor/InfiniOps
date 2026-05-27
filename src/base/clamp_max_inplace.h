#ifndef INFINI_OPS_BASE_CLAMP_MAX_INPLACE_H_
#define INFINI_OPS_BASE_CLAMP_MAX_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class ClampMaxInplace : public Operator<ClampMaxInplace> {
 public:
  ClampMaxInplace(Tensor input, const Tensor max)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor max) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
