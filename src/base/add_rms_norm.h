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
  AddRmsNorm(const Tensor input, const Tensor residual, const Tensor weight,
             float eps, Tensor out, Tensor residual_out)
      : input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        residual_strides_{residual.strides()},
        out_strides_{out.strides()},
        residual_out_strides_{residual_out.strides()},
        eps_{eps},
        dim_{out.size(-1)},
        ndim_{out.ndim()},
        batch_size_{ndim_ == 2 ? out.size(-2) : out.size(-3)},
        nhead_{ndim_ == 2 ? 1 : out.size(-2)} {
    assert(ndim_ == 2 || ndim_ == 3);
    assert(input.shape() == out.shape());
    assert(input.shape() == residual.shape());
    assert(input.shape() == residual_out.shape());
    assert(weight.ndim() == 1 && weight.size(-1) == dim_);
    assert(input.dtype() == out.dtype());
    assert(input.dtype() == residual.dtype());
    assert(input.dtype() == residual_out.dtype());
    assert(input.dtype() == weight.dtype());
    // CUDA kernel indexes the normalized dimension with stride 1.
    assert(input.stride(-1) == 1);
    assert(residual.stride(-1) == 1);
    assert(out.stride(-1) == 1);
    assert(residual_out.stride(-1) == 1);
    assert(weight.stride(-1) == 1);
  }

  virtual void operator()(const Tensor input, const Tensor residual,
                          const Tensor weight, float eps, Tensor out,
                          Tensor residual_out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides residual_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides residual_out_strides_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
