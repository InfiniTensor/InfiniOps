#ifndef INFINI_OPS_BASE_LINALG_PINV_RCOND_TENSOR_H_
#define INFINI_OPS_BASE_LINALG_PINV_RCOND_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class LinalgPinvRcondTensor : public Operator<LinalgPinvRcondTensor> {
 public:
  LinalgPinvRcondTensor(const Tensor self, const Tensor rcond, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        rcond_shape_{rcond.shape()},
        rcond_strides_{rcond.strides()},
        rcond_type_{rcond.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor rcond,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape rcond_shape_;

  Tensor::Strides rcond_strides_;

  DataType rcond_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
