#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class AddRmsNorm : public Operator<AddRmsNorm> {
 public:
  AddRmsNorm(const Tensor x1, const Tensor x2, const Tensor weight, float eps,
             Tensor y_out, Tensor x_out)
      : x1_strides_{x1.strides()},
        x2_strides_{x2.strides()},
        y_out_strides_{y_out.strides()},
        x_out_strides_{x_out.strides()},
        eps_{eps},
        dim_{y_out.size(-1)},
        ndim_{y_out.ndim()},
        batch_size_{ndim_ == 2 ? y_out.size(-2) : y_out.size(-3)},
        nhead_{ndim_ == 2 ? 1 : y_out.size(-2)} {
    assert(x1.dtype() == x2.dtype() && x1.dtype() == weight.dtype() &&
           x1.dtype() == y_out.dtype() && x1.dtype() == x_out.dtype());
  }

  virtual void operator()(const Tensor x1, const Tensor x2,
                          const Tensor weight, float eps, Tensor y_out,
                          Tensor x_out) const = 0;

 protected:
  Tensor::Strides x1_strides_;

  Tensor::Strides x2_strides_;

  Tensor::Strides y_out_strides_;

  Tensor::Strides x_out_strides_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
