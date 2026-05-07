#ifndef INFINI_OPS_BASE_RRELU_WITH_NOISE_H_
#define INFINI_OPS_BASE_RRELU_WITH_NOISE_H_

#include "operator.h"

namespace infini::ops {

class RreluWithNoise : public Operator<RreluWithNoise> {
 public:
  RreluWithNoise(const Tensor self, const Tensor noise, const double lower,
                 const double upper, Tensor out)
      : self_shape_{self.shape()},
        self_strides_{self.strides()},
        self_type_{self.dtype()},
        noise_shape_{noise.shape()},
        noise_strides_{noise.strides()},
        noise_type_{noise.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor self, const Tensor noise,
                          const double lower, const double upper,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape self_shape_;
  Tensor::Strides self_strides_;
  DataType self_type_;
  Tensor::Shape noise_shape_;
  Tensor::Strides noise_strides_;
  DataType noise_type_;
  Tensor::Shape out_shape_;
  Tensor::Strides out_strides_;
  DataType out_type_;
  int device_index_{0};
};

}  // namespace infini::ops

#endif
