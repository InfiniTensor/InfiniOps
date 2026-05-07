#ifndef INFINI_OPS_BASE_LINALG_LDL_FACTOR_H_
#define INFINI_OPS_BASE_LINALG_LDL_FACTOR_H_

#include "operator.h"

namespace infini::ops {

class LinalgLdlFactor : public Operator<LinalgLdlFactor> {
 public:
  LinalgLdlFactor(const Tensor self, Tensor LD, Tensor pivots)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        LD_shape_{LD.shape()},
        LD_strides_{LD.strides()},
        LD_type_{LD.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        device_index_{LD.device().index()} {}

  virtual void operator()(const Tensor self, Tensor LD,
                          Tensor pivots) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape LD_shape_;
  Tensor::Strides LD_strides_;
  DataType LD_type_;
  Tensor::Shape pivots_shape_;
  Tensor::Strides pivots_strides_;
  DataType pivots_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
