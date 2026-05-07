#ifndef INFINI_OPS_BASE_QR_Q_H_
#define INFINI_OPS_BASE_QR_Q_H_

#include "operator.h"

namespace infini::ops {

class QrQ : public Operator<QrQ> {
 public:
  QrQ(const Tensor self, Tensor Q, Tensor R)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        Q_shape_{Q.shape()},
        Q_strides_{Q.strides()},
        Q_type_{Q.dtype()},
        R_shape_{R.shape()},
        R_strides_{R.strides()},
        R_type_{R.dtype()},
        device_index_{Q.device().index()} {}

  virtual void operator()(const Tensor self, Tensor Q, Tensor R) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

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
