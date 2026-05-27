#ifndef INFINI_OPS_BASE_ADD_INPLACE_H_
#define INFINI_OPS_BASE_ADD_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class AddInplace : public Operator<AddInplace> {
 public:
  AddInplace(Tensor input, const Tensor other, const double alpha)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        alpha_{alpha},
        device_index_{input.device().index()} {}

  AddInplace(Tensor input, const double other, const double alpha)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        alpha_{alpha},
        other_{other},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor other,
                          const double alpha) const = 0;

  virtual void operator()(Tensor input, const double other,
                          const double alpha) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  double alpha_{};

  double other_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
