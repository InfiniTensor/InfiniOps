#ifndef INFINI_OPS_BASE_LINALG_CHOLESKY_EX_L_H_
#define INFINI_OPS_BASE_LINALG_CHOLESKY_EX_L_H_

#include "operator.h"

namespace infini::ops {

class LinalgCholeskyExL : public Operator<LinalgCholeskyExL> {
 public:
  LinalgCholeskyExL(const Tensor self, Tensor L, Tensor info)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        L_shape_{L.shape()},
        L_strides_{L.strides()},
        L_type_{L.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        device_index_{L.device().index()} {}

  virtual void operator()(const Tensor self, Tensor L, Tensor info) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape L_shape_;
  Tensor::Strides L_strides_;
  DataType L_type_;
  Tensor::Shape info_shape_;
  Tensor::Strides info_strides_;
  DataType info_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
