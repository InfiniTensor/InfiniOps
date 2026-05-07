#ifndef INFINI_OPS_BASE_LERP_SCALAR_H_
#define INFINI_OPS_BASE_LERP_SCALAR_H_

#include "operator.h"

namespace infini::ops {

class LerpScalar : public Operator<LerpScalar> {
 public:
  LerpScalar(const Tensor self, const Tensor end, const double weight,
             Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        end_shape_{end.shape()},
        end_strides_{end.strides()},
        end_type_{end.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor end,
                          const double weight, Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape end_shape_;
  Tensor::Strides end_strides_;
  DataType end_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
