#ifndef INFINI_OPS_BASE_LINALG_EIGH_EIGVALS_H_
#define INFINI_OPS_BASE_LINALG_EIGH_EIGVALS_H_

#include "operator.h"

namespace infini::ops {

class LinalgEighEigvals : public Operator<LinalgEighEigvals> {
 public:
  LinalgEighEigvals(const Tensor self, Tensor eigvals, Tensor eigvecs)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        eigvals_shape_{eigvals.shape()},
        eigvals_strides_{eigvals.strides()},
        eigvals_type_{eigvals.dtype()},
        eigvecs_shape_{eigvecs.shape()},
        eigvecs_strides_{eigvecs.strides()},
        eigvecs_type_{eigvecs.dtype()},
        device_index_{eigvals.device().index()} {}

  virtual void operator()(const Tensor self, Tensor eigvals,
                          Tensor eigvecs) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape eigvals_shape_;
  Tensor::Strides eigvals_strides_;
  DataType eigvals_type_;
  Tensor::Shape eigvecs_shape_;
  Tensor::Strides eigvecs_strides_;
  DataType eigvecs_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
