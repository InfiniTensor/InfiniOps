#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class AddRmsNorm : public Operator<AddRmsNorm> {
 public:
  AddRmsNorm(const Tensor x1, const Tensor x2, const Tensor gamma, float eps,
             Tensor y_out, Tensor x_out)
      : input_shape_{x1.shape()},
        eps_{eps},
        dim_{x1.size(-1)},
        ndim_{x1.ndim()},
        batch_size_{ndim_ == 2 ? x1.size(-2) : x1.size(-3)},
        nhead_{ndim_ == 2 ? 1 : x1.size(-2)},
        rstd_shape_{static_cast<int64_t>(batch_size_),
                    static_cast<int64_t>(nhead_)} {
    assert(x1.dtype() == x2.dtype());
    assert(x1.dtype() == y_out.dtype());
    assert(x1.dtype() == x_out.dtype());
  }

  virtual void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                          float eps, Tensor y_out, Tensor x_out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};

  std::vector<int64_t> rstd_shape_;
};

}  // namespace infini::ops

#endif
