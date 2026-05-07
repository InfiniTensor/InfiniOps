#ifndef INFINI_OPS_BASE_CLAMP_MAX_TENSOR_H_
#define INFINI_OPS_BASE_CLAMP_MAX_TENSOR_H_

#include "operator.h"

namespace infini::ops {

class ClampMaxTensor : public Operator<ClampMaxTensor> {
 public:
  ClampMaxTensor(const Tensor self, const Tensor max, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor max,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;

  Tensor::Strides self_strides_;

  DataType self_type_;

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
