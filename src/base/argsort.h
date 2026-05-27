#ifndef INFINI_OPS_BASE_ARGSORT_H_
#define INFINI_OPS_BASE_ARGSORT_H_

#include "operator.h"

namespace infini::ops {

class Argsort : public Operator<Argsort> {
 public:
  Argsort(const Tensor input, const bool stable, const int64_t dim,
          const bool descending, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        stable_{stable},
        dim_{dim},
        descending_{descending},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const bool stable,
                          const int64_t dim, const bool descending,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool stable_{};

  int64_t dim_{};

  bool descending_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
