#ifndef INFINI_OPS_BASE_MAXIMUM_H_
#define INFINI_OPS_BASE_MAXIMUM_H_

#include "operator.h"

namespace infini::ops {

class Maximum : public Operator<Maximum> {
 public:
  Maximum(const Tensor self, const Tensor other, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor other,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape other_shape_;
  Tensor::Strides other_strides_;
  DataType other_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
