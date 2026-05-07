#ifndef INFINI_OPS_BASE_AMINMAX_H_
#define INFINI_OPS_BASE_AMINMAX_H_

#include "operator.h"

namespace infini::ops {

class Aminmax : public Operator<Aminmax> {
 public:
  Aminmax(const Tensor self, Tensor min, Tensor max)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        device_index_{min.device().index()} {}

  virtual void operator()(const Tensor self, Tensor min, Tensor max) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape min_shape_;
  Tensor::Strides min_strides_;
  DataType min_type_;
  Tensor::Shape max_shape_;
  Tensor::Strides max_strides_;
  DataType max_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
