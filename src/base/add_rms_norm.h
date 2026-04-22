#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// vLLM-aligned fused add + RMSNorm.
//
// Mirrors `vllm._C.fused_add_rms_norm(input, residual, weight, eps)`
// semantics: `residual_out = input + residual`; `out = RMSNorm(residual_out)
// * weight`.  vLLM's schema is fully inplace (both `input` and `residual` are
// written back); this base exposes an out-of-place primary form (matches the
// project's C2 "inputs-first-outputs-last" rule and supports callers that
// need separate destinations) plus an inplace convenience overload that
// routes to the out-of-place form with aliased buffers.
class AddRmsNorm : public Operator<AddRmsNorm> {
 public:
  AddRmsNorm(const Tensor input, const Tensor residual, const Tensor weight,
             float eps, Tensor out, Tensor residual_out)
      : input_shape_{input.shape()},
        eps_{eps},
        dim_{input.size(-1)},
        ndim_{input.ndim()},
        batch_size_{ndim_ == 2 ? input.size(-2) : input.size(-3)},
        nhead_{ndim_ == 2 ? 1 : input.size(-2)} {
    assert(input.dtype() == residual.dtype() &&
           "`AddRmsNorm`: `input` and `residual` must have the same dtype.");
    assert(input.dtype() == out.dtype() &&
           "`AddRmsNorm`: `input` and `out` must have the same dtype.");
    assert(
        input.dtype() == residual_out.dtype() &&
        "`AddRmsNorm`: `input` and `residual_out` must have the same dtype.");
  }

  // Inplace convenience overload — matches vLLM's
  // `fused_add_rms_norm(input, residual, weight, eps)` signature directly.
  // Aliases `out = input` and `residual_out = residual`; callers that need
  // separate destinations should use the primary constructor.
  AddRmsNorm(Tensor input, Tensor residual, const Tensor weight, float eps)
      : AddRmsNorm{input, residual, weight, eps, input, residual} {}

  virtual void operator()(const Tensor input, const Tensor residual,
                          const Tensor weight, float eps, Tensor out,
                          Tensor residual_out) const = 0;

  virtual void operator()(Tensor input, Tensor residual, const Tensor weight,
                          float eps) const {
    return operator()(input, residual, weight, eps, input, residual);
  }

 protected:
  Tensor::Shape input_shape_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
