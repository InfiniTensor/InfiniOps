#ifndef INFINI_OPS_BASE_SVD_U_H_
#define INFINI_OPS_BASE_SVD_U_H_

#include "operator.h"

namespace infini::ops {

class SvdU : public Operator<SvdU> {
 public:
  SvdU(const Tensor self, Tensor U, Tensor S, Tensor V)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        U_shape_{U.shape()},
        U_strides_{U.strides()},
        U_type_{U.dtype()},
        S_shape_{S.shape()},
        S_strides_{S.strides()},
        S_type_{S.dtype()},
        V_shape_{V.shape()},
        V_strides_{V.strides()},
        V_type_{V.dtype()},
        device_index_{U.device().index()} {}

  virtual void operator()(const Tensor self, Tensor U, Tensor S,
                          Tensor V) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape U_shape_;

  Tensor::Strides U_strides_;

  DataType U_type_;

  Tensor::Shape S_shape_;

  Tensor::Strides S_strides_;

  DataType S_type_;

  Tensor::Shape V_shape_;

  Tensor::Strides V_strides_;

  DataType V_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
