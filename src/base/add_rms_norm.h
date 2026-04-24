#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class AddRmsNorm : public Operator<AddRmsNorm> {
 public:
  // TODO: Make `eps` an `std::optional<float>` with a PyTorch-aligned default.
  // Also consider the same change for `RmsNorm`.
  AddRmsNorm(const Tensor input, const Tensor other, const Tensor weight,
             float eps, Tensor out, Tensor rstd_out)
      : input_shape_{input.shape()},
        eps_{eps},
        dim_{input.size(-1)},
        ndim_{input.ndim()},
        batch_size_{ndim_ == 2 ? input.size(-2) : input.size(-3)},
        nhead_{ndim_ == 2 ? 1 : input.size(-2)},
        rstd_shape_{static_cast<int64_t>(batch_size_),
                    static_cast<int64_t>(nhead_)} {
    assert(input.dtype() == other.dtype());
    assert(input.dtype() == out.dtype());
    assert(input.dtype() == rstd_out.dtype());
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          const Tensor weight, float eps, Tensor out,
                          Tensor rstd_out) const = 0;

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
