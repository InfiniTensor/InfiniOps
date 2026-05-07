#ifndef INFINI_OPS_BASE_LINALG_LU_FACTOR_H_
#define INFINI_OPS_BASE_LINALG_LU_FACTOR_H_

#include "operator.h"

namespace infini::ops {

class LinalgLuFactor : public Operator<LinalgLuFactor> {
 public:
  LinalgLuFactor(const Tensor A, Tensor LU, Tensor pivots)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        device_index_{LU.device().index()} {}

  virtual void operator()(const Tensor A, Tensor LU, Tensor pivots) const = 0;

 protected:
  Tensor::Shape A_shape_;
  Tensor::Strides A_strides_;
  DataType A_type_;
  Tensor::Shape LU_shape_;
  Tensor::Strides LU_strides_;
  DataType LU_type_;
  Tensor::Shape pivots_shape_;
  Tensor::Strides pivots_strides_;
  DataType pivots_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
