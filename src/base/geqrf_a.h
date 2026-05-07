#ifndef INFINI_OPS_BASE_GEQRF_A_H_
#define INFINI_OPS_BASE_GEQRF_A_H_

#include "operator.h"

namespace infini::ops {

class GeqrfA : public Operator<GeqrfA> {
 public:
  GeqrfA(const Tensor self, Tensor a, Tensor tau)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        a_shape_{a.shape()},
        a_strides_{a.strides()},
        a_type_{a.dtype()},
        tau_shape_{tau.shape()},
        tau_strides_{tau.strides()},
        tau_type_{tau.dtype()},
        device_index_{a.device().index()} {}

  virtual void operator()(const Tensor self, Tensor a, Tensor tau) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape a_shape_;
  Tensor::Strides a_strides_;
  DataType a_type_;
  Tensor::Shape tau_shape_;
  Tensor::Strides tau_strides_;
  DataType tau_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
