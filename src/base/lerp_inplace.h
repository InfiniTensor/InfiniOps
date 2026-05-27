#ifndef INFINI_OPS_BASE_LERP_INPLACE_H_
#define INFINI_OPS_BASE_LERP_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class LerpInplace : public Operator<LerpInplace> {
 public:
  LerpInplace(Tensor input, const Tensor end, const double weight)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        end_shape_{end.shape()},
        end_strides_{end.strides()},
        end_type_{end.dtype()},
        weight_{weight},
        device_index_{input.device().index()} {}

  LerpInplace(Tensor input, const Tensor end, const Tensor weight)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        end_shape_{end.shape()},
        end_strides_{end.strides()},
        end_type_{end.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor end,
                          const double weight) const = 0;

  virtual void operator()(Tensor input, const Tensor end,
                          const Tensor weight) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape end_shape_;

  Tensor::Strides end_strides_;

  DataType end_type_;

  double weight_{};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
