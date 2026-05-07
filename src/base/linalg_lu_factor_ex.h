#ifndef INFINI_OPS_BASE_LINALG_LU_FACTOR_EX_H_
#define INFINI_OPS_BASE_LINALG_LU_FACTOR_EX_H_

#include "operator.h"

namespace infini::ops {

class LinalgLuFactorEx : public Operator<LinalgLuFactorEx> {
 public:
  LinalgLuFactorEx(const Tensor A, Tensor LU, Tensor pivots, Tensor info)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        device_index_{LU.device().index()} {}

  virtual void operator()(const Tensor A, Tensor LU, Tensor pivots,
                          Tensor info) const = 0;

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
  Tensor::Shape info_shape_;
  Tensor::Strides info_strides_;
  DataType info_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
