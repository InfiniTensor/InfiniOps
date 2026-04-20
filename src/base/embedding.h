#ifndef INFINI_OPS_BASE_EMBEDDING_H_
#define INFINI_OPS_BASE_EMBEDDING_H_

#include "operator.h"

namespace infini::ops {

class Embedding : public Operator<Embedding> {
 public:
  Embedding(const Tensor input, const Tensor weight, Tensor out)
      : input_type_{input.dtype()},
        weight_type_{weight.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        weight_shape_{weight.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        weight_strides_{weight.strides()},
        out_strides_{out.strides()} {
    assert(weight.ndim() == 2 && "`weight` must be 2D");
    assert(weight.dtype() == out.dtype() &&
           "`weight` and `out` must have the same dtype");
    assert(out.ndim() == input.ndim() + 1 &&
           "`out.ndim()` must equal `input.ndim() + 1`");
    assert(out.size(-1) == weight.size(-1) &&
           "`out.size(-1)` must equal `weight.size(-1)`");
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          Tensor out) const = 0;

 protected:
  const DataType input_type_;

  const DataType weight_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape weight_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides weight_strides_;

  Tensor::Strides out_strides_;
};

}  // namespace infini::ops

#endif
