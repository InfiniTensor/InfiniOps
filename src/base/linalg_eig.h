#ifndef INFINI_OPS_BASE_LINALG_EIG_H_
#define INFINI_OPS_BASE_LINALG_EIG_H_

#include "operator.h"

namespace infini::ops {

class LinalgEig : public Operator<LinalgEig> {
 public:
  LinalgEig(const Tensor self, Tensor eigenvalues, Tensor eigenvectors)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        eigenvalues_shape_{eigenvalues.shape()},
        eigenvalues_strides_{eigenvalues.strides()},
        eigenvalues_type_{eigenvalues.dtype()},
        eigenvectors_shape_{eigenvectors.shape()},
        eigenvectors_strides_{eigenvectors.strides()},
        eigenvectors_type_{eigenvectors.dtype()},
        device_index_{eigenvalues.device().index()} {}

  virtual void operator()(const Tensor self, Tensor eigenvalues,
                          Tensor eigenvectors) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape eigenvalues_shape_;
  Tensor::Strides eigenvalues_strides_;
  DataType eigenvalues_type_;
  Tensor::Shape eigenvectors_shape_;
  Tensor::Strides eigenvectors_strides_;
  DataType eigenvectors_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
