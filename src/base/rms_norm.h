#ifndef INFINI_OPS_BASE_RMS_NORM_H_
#define INFINI_OPS_BASE_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class RmsNorm : public Operator<RmsNorm> {
 public:
  RmsNorm(const Tensor y, const Tensor x, const Tensor w, float epsilon)
      : epsilon_{epsilon},
        y_shape_{y.shape()},
        x_shape_{x.shape()},
        y_strides_{y.strides()},
        x_strides_{x.strides()},
        dim_{y.size(-1)},
        ndim_{y.ndim()},
        batch_size_{ndim_ == 2 ? y.size(-2) : y.size(-3)},
        nhead_{ndim_ == 2 ? 1 : y.size(-2)} {}

  RmsNorm(const Tensor y, const Tensor x, const Tensor w)
      : RmsNorm{y, x, w, 1e-6f} {}

  virtual void operator()(void* stream, Tensor y, const Tensor x,
                          const Tensor w, float epsilon) const = 0;

  virtual void operator()(void* stream, Tensor y, const Tensor x,
                          const Tensor w) const {
    return operator()(stream, y, x, w, epsilon_);
  }

 protected:
  float epsilon_{1e-6f};

  Tensor::Shape y_shape_;

  Tensor::Shape x_shape_;

  Tensor::Strides y_strides_;

  Tensor::Strides x_strides_;

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
