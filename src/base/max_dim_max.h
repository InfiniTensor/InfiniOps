#ifndef INFINI_OPS_BASE_MAX_DIM_MAX_H_
#define INFINI_OPS_BASE_MAX_DIM_MAX_H_

#include "operator.h"

namespace infini::ops {

class MaxDimMax : public Operator<MaxDimMax> {
 public:
  MaxDimMax(const Tensor self, const int64_t dim, Tensor max, Tensor max_values)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        max_values_shape_{max_values.shape()},
        max_values_strides_{max_values.strides()},
        max_values_type_{max_values.dtype()},
        device_index_{max.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t dim, Tensor max,
                          Tensor max_values) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  Tensor::Shape max_values_shape_;

  Tensor::Strides max_values_strides_;

  DataType max_values_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
