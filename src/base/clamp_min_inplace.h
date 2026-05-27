#ifndef INFINI_OPS_BASE_CLAMP_MIN_INPLACE_H_
#define INFINI_OPS_BASE_CLAMP_MIN_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class ClampMinInplace : public Operator<ClampMinInplace> {
 public:
  ClampMinInplace(Tensor input, const Tensor min)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor min) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
