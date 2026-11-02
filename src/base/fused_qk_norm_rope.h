#ifndef INFINI_OPS_BASE_FUSED_QK_NORM_ROPE_H_
#define INFINI_OPS_BASE_FUSED_QK_NORM_ROPE_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class FusedQkNormRope : public Operator<FusedQkNormRope> {
 public:
  FusedQkNormRope(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                  int64_t num_heads_v, int64_t head_dim, float eps,
                  Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache,
                  bool is_neox, Tensor position_ids,
                  int64_t forced_token_heads_per_warp = -1)
      : qkv_shape_{qkv.shape()},
        qkv_strides_{qkv.strides()},
        q_weight_shape_{q_weight.shape()},
        k_weight_shape_{k_weight.shape()},
        cos_sin_cache_shape_{cos_sin_cache.shape()},
        position_ids_shape_{position_ids.shape()},
        num_heads_q_{num_heads_q},
        num_heads_k_{num_heads_k},
        num_heads_v_{num_heads_v},
        head_dim_{head_dim},
        eps_{eps},
        rotary_dim_{static_cast<int64_t>(cos_sin_cache.size(1))},
        is_neox_{is_neox},
        forced_token_heads_per_warp_{forced_token_heads_per_warp} {
    assert(qkv.ndim() == 2 && qkv.stride(1) == 1 &&
           "`FusedQkNormRope` requires a contiguous rank-2 `qkv`");
    assert(qkv.size(1) ==
               static_cast<Tensor::Size>(
                   (num_heads_q + num_heads_k + num_heads_v) * head_dim) &&
           "`FusedQkNormRope` requires `qkv` to contain Q, K, and V");
    assert(qkv.size(0) == position_ids.size(0) &&
           "`FusedQkNormRope` requires matching token counts");
    assert(q_weight.ndim() == 1 && q_weight.size(0) == head_dim &&
           q_weight.stride(0) == 1 &&
           "`FusedQkNormRope` requires contiguous `q_weight[head_dim]`");
    assert(k_weight.ndim() == 1 && k_weight.size(0) == head_dim &&
           k_weight.stride(0) == 1 &&
           "`FusedQkNormRope` requires contiguous `k_weight[head_dim]`");
    assert(position_ids.ndim() == 1 && position_ids.stride(0) == 1 &&
           position_ids.dtype() == DataType::kInt64 &&
           "`FusedQkNormRope` requires contiguous int64 `position_ids`");
    assert(cos_sin_cache.ndim() == 2 && cos_sin_cache.stride(1) == 1 &&
           "`FusedQkNormRope` requires a contiguous rank-2 `cos_sin_cache`");
    assert(rotary_dim_ % 2 == 0 && rotary_dim_ <= head_dim &&
           "`FusedQkNormRope` requires an even rotary dimension no greater "
           "than `head_dim`");
    assert(qkv.dtype() == q_weight.dtype() &&
           q_weight.dtype() == k_weight.dtype() &&
           "`FusedQkNormRope` requires matching QKV and weight dtypes");
    assert(qkv.device() == q_weight.device() &&
           qkv.device() == k_weight.device() &&
           qkv.device() == cos_sin_cache.device() &&
           qkv.device() == position_ids.device() &&
           "`FusedQkNormRope` requires tensors on one device");
  }

  virtual void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                          int64_t num_heads_v, int64_t head_dim, float eps,
                          Tensor q_weight, Tensor k_weight,
                          Tensor cos_sin_cache, bool is_neox,
                          Tensor position_ids,
                          int64_t forced_token_heads_per_warp) const = 0;

 protected:
  Tensor::Shape qkv_shape_;

  Tensor::Strides qkv_strides_;

  Tensor::Shape q_weight_shape_;

  Tensor::Shape k_weight_shape_;

  Tensor::Shape cos_sin_cache_shape_;

  Tensor::Shape position_ids_shape_;

  int64_t num_heads_q_{0};

  int64_t num_heads_k_{0};

  int64_t num_heads_v_{0};

  int64_t head_dim_{0};

  float eps_{};

  int64_t rotary_dim_{0};

  bool is_neox_{false};

  int64_t forced_token_heads_per_warp_{-1};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FUSED_QK_NORM_ROPE_H_
