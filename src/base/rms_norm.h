#ifndef INFINI_OPS_BASE_RMS_NORM_H_
#define INFINI_OPS_BASE_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class RmsNorm : public Operator<RmsNorm> {
 public:
  // Parameter order and naming follow PyTorch: input, weight, eps, out.
  RmsNorm(const Tensor out, const Tensor input, const Tensor weight, float eps)
      : eps_{eps},
        out_shape_{out.shape()},
        input_shape_{input.shape()},
        out_strides_{out.strides()},
        input_strides_{input.strides()},
        dim_{out.size(-1)},
        ndim_{out.ndim()},
        batch_size_{ndim_ == 2 ? out.size(-2) : out.size(-3)},
        nhead_{ndim_ == 2 ? 1 : out.size(-2)} {}

  RmsNorm(const Tensor out, const Tensor input, const Tensor weight)
      : RmsNorm{out, input, weight, 1e-6f} {}

  virtual void operator()(void* stream, Tensor out, const Tensor input,
                          const Tensor weight, float eps) const = 0;

  virtual void operator()(void* stream, Tensor out, const Tensor input,
                          const Tensor weight) const {
    return operator()(stream, out, input, weight, eps_);
  }

 protected:
  float eps_{1e-6f};

  Tensor::Shape out_shape_;

  Tensor::Shape input_shape_;

  Tensor::Strides out_strides_;

  Tensor::Strides input_strides_;

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
