#ifndef INFINI_OPS_BASE_LINALG_LU_H_
#define INFINI_OPS_BASE_LINALG_LU_H_

#include "operator.h"

namespace infini::ops {

class LinalgLu : public Operator<LinalgLu> {
 public:
  LinalgLu(const Tensor A, Tensor P, Tensor L, Tensor U)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        P_shape_{P.shape()},
        P_strides_{P.strides()},
        P_type_{P.dtype()},
        L_shape_{L.shape()},
        L_strides_{L.strides()},
        L_type_{L.dtype()},
        U_shape_{U.shape()},
        U_strides_{U.strides()},
        U_type_{U.dtype()},
        device_index_{P.device().index()} {}

  virtual void operator()(const Tensor A, Tensor P, Tensor L,
                          Tensor U) const = 0;

 protected:
  Tensor::Shape A_shape_;
  Tensor::Strides A_strides_;
  DataType A_type_;
  Tensor::Shape P_shape_;
  Tensor::Strides P_strides_;
  DataType P_type_;
  Tensor::Shape L_shape_;
  Tensor::Strides L_strides_;
  DataType L_type_;
  Tensor::Shape U_shape_;
  Tensor::Strides U_strides_;
  DataType U_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
