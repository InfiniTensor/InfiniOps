#ifndef INFINI_OPS_BASE_LINALG_MATRIX_RANK_H_
#define INFINI_OPS_BASE_LINALG_MATRIX_RANK_H_

#include "operator.h"

namespace infini::ops {

class LinalgMatrixRank : public Operator<LinalgMatrixRank> {
 public:
  LinalgMatrixRank(const Tensor input, const double tol, const bool hermitian,
                   Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        tol_{tol},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  LinalgMatrixRank(const Tensor input, const bool hermitian, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  LinalgMatrixRank(const Tensor input, const Tensor tol, const bool hermitian,
                   Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        tol_shape_{tol.shape()},
        tol_strides_{tol.strides()},
        tol_type_{tol.dtype()},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double tol,
                          const bool hermitian, Tensor out) const = 0;

  virtual void operator()(const Tensor input, const bool hermitian,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor tol,
                          const bool hermitian, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double tol_{};

  bool hermitian_{};

  Tensor::Shape tol_shape_;

  Tensor::Strides tol_strides_;

  DataType tol_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
