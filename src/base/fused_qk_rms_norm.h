#ifndef INFINI_OPS_BASE_FUSED_QK_RMS_NORM_H_
#define INFINI_OPS_BASE_FUSED_QK_RMS_NORM_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class FusedQkRmsNorm : public Operator<FusedQkRmsNorm> {
 public:
  FusedQkRmsNorm(Tensor q, Tensor kv, Tensor q_weight, Tensor kv_weight,
                 float eps, Tensor q_output, Tensor kv_output)
      : q_shape_{q.shape()},
        kv_shape_{kv.shape()},
        q_output_shape_{q_output.shape()},
        kv_output_shape_{kv_output.shape()},
        q_weight_shape_{q_weight.shape()},
        kv_weight_shape_{kv_weight.shape()},
        eps_{eps} {
    assert(q.ndim() == 3 && q.stride(2) == 1 &&
           "`FusedQkRmsNorm` requires rank-3 `q` with a contiguous last dim");
    assert(kv.ndim() == 3 && kv.stride(2) == 1 &&
           "`FusedQkRmsNorm` requires rank-3 `kv` with a contiguous last dim");
    assert(q.size(0) == kv.size(0) &&
           "`FusedQkRmsNorm` requires matching token counts");
    // The kernel walks the outputs with their own token and head strides
    // too, so only the last dim must be contiguous.
    assert(q_output.ndim() == 3 && q_output.shape() == q.shape() &&
           q_output.stride(2) == 1 &&
           "`FusedQkRmsNorm` requires a rank-3 `q_output` matching `q` with a "
           "contiguous last dim");
    assert(kv_output.ndim() == 3 && kv_output.shape() == kv.shape() &&
           kv_output.stride(2) == 1 &&
           "`FusedQkRmsNorm` requires a rank-3 `kv_output` matching `kv` with "
           "a contiguous last dim");
    assert(q_weight.ndim() == 1 && q_weight.size(0) == q.size(2) &&
           q_weight.stride(0) == 1 &&
           "`FusedQkRmsNorm` requires contiguous `q_weight[head_dim]`");
    assert(kv_weight.ndim() == 1 && kv_weight.size(0) == kv.size(2) &&
           kv_weight.stride(0) == 1 &&
           "`FusedQkRmsNorm` requires contiguous `kv_weight[head_dim]`");
    assert(q.dtype() == q_weight.dtype() &&
           q_weight.dtype() == kv_weight.dtype() &&
           kv_weight.dtype() == q_output.dtype() &&
           q_output.dtype() == kv_output.dtype() &&
           "`FusedQkRmsNorm` requires matching tensor dtypes");
    assert(
        q.device() == q_weight.device() && q.device() == kv_weight.device() &&
        q.device() == q_output.device() && q.device() == kv_output.device() &&
        "`FusedQkRmsNorm` requires tensors on one device");
  }

  virtual void operator()(Tensor q, Tensor kv, Tensor q_weight,
                          Tensor kv_weight, float eps, Tensor q_output,
                          Tensor kv_output) const = 0;

 protected:
  Tensor::Shape q_shape_;

  Tensor::Shape kv_shape_;

  Tensor::Shape q_output_shape_;

  Tensor::Shape kv_output_shape_;

  Tensor::Shape q_weight_shape_;

  Tensor::Shape kv_weight_shape_;

  float eps_{};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FUSED_QK_RMS_NORM_H_
