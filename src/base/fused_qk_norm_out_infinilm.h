#ifndef INFINI_OPS_BASE_FUSED_QK_NORM_OUT_INFINILM_H_
#define INFINI_OPS_BASE_FUSED_QK_NORM_OUT_INFINILM_H_

#include <cassert>
#include <cstddef>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class FusedQkNormOutInfinilm : public Operator<FusedQkNormOutInfinilm> {
 public:
  FusedQkNormOutInfinilm(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                         int64_t num_heads_v, int64_t head_dim, float eps,
                         Tensor q_weight, Tensor k_weight, Tensor q_output,
                         Tensor k_output)
      : qkv_shape_{qkv.shape()},
        q_output_shape_{q_output.shape()},
        k_output_shape_{k_output.shape()},
        q_weight_shape_{q_weight.shape()},
        k_weight_shape_{k_weight.shape()},
        num_heads_q_{num_heads_q},
        num_heads_k_{num_heads_k},
        num_heads_v_{num_heads_v},
        head_dim_{head_dim},
        eps_{eps} {
    [[maybe_unused]] const auto q_width =
        static_cast<Tensor::Size>(num_heads_q * head_dim);
    [[maybe_unused]] const auto k_width =
        static_cast<Tensor::Size>(num_heads_k * head_dim);
    [[maybe_unused]] const auto total_width = static_cast<Tensor::Size>(
        (num_heads_q + num_heads_k + num_heads_v) * head_dim);
    assert(qkv.ndim() == 2 && qkv.IsContiguous() &&
           "`FusedQkNormOutInfinilm` requires a contiguous rank-2 `qkv`");
    assert(qkv.size(1) == total_width &&
           "`FusedQkNormOutInfinilm` requires `qkv` to contain Q, K, and V");
    assert(
        q_output.ndim() == 2 && q_output.IsContiguous() &&
        q_output.size(0) == qkv.size(0) && q_output.size(1) == q_width &&
        "`FusedQkNormOutInfinilm` requires a matching contiguous `q_output`");
    assert(
        k_output.ndim() == 2 && k_output.IsContiguous() &&
        k_output.size(0) == qkv.size(0) && k_output.size(1) == k_width &&
        "`FusedQkNormOutInfinilm` requires a matching contiguous `k_output`");
    assert(q_weight.ndim() == 1 && q_weight.size(0) == head_dim &&
           q_weight.stride(0) == 1 &&
           "`FusedQkNormOutInfinilm` requires contiguous `q_weight[head_dim]`");
    assert(k_weight.ndim() == 1 && k_weight.size(0) == head_dim &&
           k_weight.stride(0) == 1 &&
           "`FusedQkNormOutInfinilm` requires contiguous `k_weight[head_dim]`");
    assert(qkv.dtype() == q_weight.dtype() &&
           q_weight.dtype() == k_weight.dtype() &&
           q_weight.dtype() == q_output.dtype() &&
           q_output.dtype() == k_output.dtype() &&
           "`FusedQkNormOutInfinilm` requires matching tensor dtypes");
    assert(qkv.device() == q_weight.device() &&
           qkv.device() == k_weight.device() &&
           qkv.device() == q_output.device() &&
           qkv.device() == k_output.device() &&
           "`FusedQkNormOutInfinilm` requires tensors on one device");
  }

  virtual void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                          int64_t num_heads_v, int64_t head_dim, float eps,
                          Tensor q_weight, Tensor k_weight, Tensor q_output,
                          Tensor k_output) const = 0;

 protected:
  Tensor::Shape qkv_shape_;

  Tensor::Shape q_output_shape_;

  Tensor::Shape k_output_shape_;

  Tensor::Shape q_weight_shape_;

  Tensor::Shape k_weight_shape_;

  int64_t num_heads_q_{0};

  int64_t num_heads_k_{0};

  int64_t num_heads_v_{0};

  int64_t head_dim_{0};

  float eps_{};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FUSED_QK_NORM_OUT_INFINILM_H_
