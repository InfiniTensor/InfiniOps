#ifndef INFINI_OPS_BASE_TRIANGULAR_SOLVE_X_H_
#define INFINI_OPS_BASE_TRIANGULAR_SOLVE_X_H_

#include "operator.h"

namespace infini::ops {

class TriangularSolveX : public Operator<TriangularSolveX> {
 public:
  TriangularSolveX(const Tensor self, const Tensor A, Tensor X, Tensor M)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        X_shape_{X.shape()},
        X_strides_{X.strides()},
        X_type_{X.dtype()},
        M_shape_{M.shape()},
        M_strides_{M.strides()},
        M_type_{M.dtype()},
        device_index_{X.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor A, Tensor X,
                          Tensor M) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape X_shape_;

  Tensor::Strides X_strides_;

  DataType X_type_;

  Tensor::Shape M_shape_;

  Tensor::Strides M_strides_;

  DataType M_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
