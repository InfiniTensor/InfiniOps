#ifndef INFINI_OPS_BASE_MAT_MUL_H_
#define INFINI_OPS_BASE_MAT_MUL_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class MatMul : public Operator<MatMul> {
 public:
  MatMul(const Tensor input, const Tensor other, Tensor out)
      : input_shape_{input.shape()},
        other_shape_{other.shape()},
        out_shape_{out.shape()} {
    assert(input.dtype() == other.dtype());
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Shape other_shape_;

  Tensor::Shape out_shape_;
};

}  // namespace infini::ops

#endif
