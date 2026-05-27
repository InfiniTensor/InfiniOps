#ifndef INFINI_OPS_BASE_BERNOULLI_INPLACE_H_
#define INFINI_OPS_BASE_BERNOULLI_INPLACE_H_

#include "operator.h"

namespace infini::ops {

class BernoulliInplace : public Operator<BernoulliInplace> {
 public:
  BernoulliInplace(Tensor input, const Tensor p)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        p_shape_{p.shape()},
        p_strides_{p.strides()},
        p_type_{p.dtype()},
        device_index_{input.device().index()} {}

  BernoulliInplace(Tensor input, const double p)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        p_{p},
        device_index_{input.device().index()} {}

  virtual void operator()(Tensor input, const Tensor p) const = 0;

  virtual void operator()(Tensor input, const double p) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape p_shape_;

  Tensor::Strides p_strides_;

  DataType p_type_;

  double p_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
