#ifndef INFINI_OPS_BASE_LINALG_LSTSQ_H_
#define INFINI_OPS_BASE_LINALG_LSTSQ_H_

#include "operator.h"

namespace infini::ops {

class LinalgLstsq : public Operator<LinalgLstsq> {
 public:
  LinalgLstsq(const Tensor self, const Tensor b, Tensor solution,
              Tensor residuals, Tensor rank, Tensor singular_values)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        b_shape_{b.shape()},
        b_strides_{b.strides()},
        b_type_{b.dtype()},
        solution_shape_{solution.shape()},
        solution_strides_{solution.strides()},
        solution_type_{solution.dtype()},
        residuals_shape_{residuals.shape()},
        residuals_strides_{residuals.strides()},
        residuals_type_{residuals.dtype()},
        rank_shape_{rank.shape()},
        rank_strides_{rank.strides()},
        rank_type_{rank.dtype()},
        singular_values_shape_{singular_values.shape()},
        singular_values_strides_{singular_values.strides()},
        singular_values_type_{singular_values.dtype()},
        device_index_{solution.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor b, Tensor solution,
                          Tensor residuals, Tensor rank,
                          Tensor singular_values) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape b_shape_;
  Tensor::Strides b_strides_;
  DataType b_type_;
  Tensor::Shape solution_shape_;
  Tensor::Strides solution_strides_;
  DataType solution_type_;
  Tensor::Shape residuals_shape_;
  Tensor::Strides residuals_strides_;
  DataType residuals_type_;
  Tensor::Shape rank_shape_;
  Tensor::Strides rank_strides_;
  DataType rank_type_;
  Tensor::Shape singular_values_shape_;
  Tensor::Strides singular_values_strides_;
  DataType singular_values_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
