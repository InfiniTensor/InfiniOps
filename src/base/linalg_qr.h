#ifndef INFINI_OPS_BASE_LINALG_QR_H_
#define INFINI_OPS_BASE_LINALG_QR_H_

#include "operator.h"

namespace infini::ops {

class LinalgQr : public Operator<LinalgQr> {
 public:
  LinalgQr(const Tensor A, Tensor Q, Tensor R)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        Q_shape_{Q.shape()},
        Q_strides_{Q.strides()},
        Q_type_{Q.dtype()},
        R_shape_{R.shape()},
        R_strides_{R.strides()},
        R_type_{R.dtype()},
        device_index_{Q.device().index()} {}

  virtual void operator()(const Tensor A, Tensor Q, Tensor R) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape Q_shape_;

  Tensor::Strides Q_strides_;

  DataType Q_type_;

  Tensor::Shape R_shape_;

  Tensor::Strides R_strides_;

  DataType R_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
