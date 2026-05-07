#ifndef INFINI_OPS_BASE_MIN_DIM_MIN_H_
#define INFINI_OPS_BASE_MIN_DIM_MIN_H_

#include "operator.h"

namespace infini::ops {

class MinDimMin : public Operator<MinDimMin> {
 public:
  MinDimMin(const Tensor self, const int64_t dim, Tensor min,
            Tensor min_indices)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        min_indices_shape_{min_indices.shape()},
        min_indices_strides_{min_indices.strides()},
        min_indices_type_{min_indices.dtype()},
        device_index_{min.device().index()} {}

  virtual void operator()(const Tensor self, const int64_t dim, Tensor min,
                          Tensor min_indices) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_;

  Tensor::Shape min_indices_shape_;

  Tensor::Strides min_indices_strides_;

  DataType min_indices_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
