#ifndef INFINI_OPS_BASE_FUSED_QK_NORM_H_
#define INFINI_OPS_BASE_FUSED_QK_NORM_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class FusedQkNorm : public Operator<FusedQkNorm> {
 public:
  FusedQkNorm(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
              int64_t num_heads_v, int64_t head_dim, float eps, Tensor q_weight,
              Tensor k_weight)
      : qkv_shape_{qkv.shape()},
        q_weight_shape_{q_weight.shape()},
        k_weight_shape_{k_weight.shape()},
        num_heads_q_{num_heads_q},
        num_heads_k_{num_heads_k},
        num_heads_v_{num_heads_v},
        head_dim_{head_dim},
        eps_{eps} {
    assert(qkv.ndim() == 2 && qkv.stride(1) == 1 &&
           "`FusedQkNorm` requires a contiguous rank-2 `qkv`");
    assert(qkv.size(1) ==
               static_cast<Tensor::Size>(
                   (num_heads_q + num_heads_k + num_heads_v) * head_dim) &&
           "`FusedQkNorm` requires `qkv` to contain Q, K, and V");
    assert(q_weight.ndim() == 1 && q_weight.size(0) == head_dim &&
           q_weight.stride(0) == 1 &&
           "`FusedQkNorm` requires contiguous `q_weight[head_dim]`");
    assert(k_weight.ndim() == 1 && k_weight.size(0) == head_dim &&
           k_weight.stride(0) == 1 &&
           "`FusedQkNorm` requires contiguous `k_weight[head_dim]`");
    assert(qkv.dtype() == q_weight.dtype() &&
           q_weight.dtype() == k_weight.dtype() &&
           "`FusedQkNorm` requires matching tensor dtypes");
    assert(qkv.device() == q_weight.device() &&
           qkv.device() == k_weight.device() &&
           "`FusedQkNorm` requires tensors on one device");
  }

  virtual void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                          int64_t num_heads_v, int64_t head_dim, float eps,
                          Tensor q_weight, Tensor k_weight) const = 0;

 protected:
  Tensor::Shape qkv_shape_;

  Tensor::Shape q_weight_shape_;

  Tensor::Shape k_weight_shape_;

  int64_t num_heads_q_{0};

  int64_t num_heads_k_{0};

  int64_t num_heads_v_{0};

  int64_t head_dim_{0};

  float eps_{};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FUSED_QK_NORM_H_
