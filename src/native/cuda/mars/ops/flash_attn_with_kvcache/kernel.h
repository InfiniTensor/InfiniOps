#ifndef INFINI_OPS_MARS_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
#define INFINI_OPS_MARS_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_

#include <optional>
#include <vector>

#include "base/flash_attn_with_kvcache.h"
#include "native/cuda/mars/ops/flash_attn_common.h"

namespace infini::ops {

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kMars, 0>
    : public FlashAttnWithKvcache {
 public:
  Operator(const Tensor q, Tensor k_cache, Tensor v_cache,
           const std::optional<Tensor> k, const std::optional<Tensor> v,
           const std::optional<Tensor> rotary_cos,
           const std::optional<Tensor> rotary_sin,
           const std::optional<Tensor> cache_seqlens,
           const std::optional<Tensor> cache_batch_idx,
           const std::optional<Tensor> cache_leftpad,
           const std::optional<Tensor> block_table,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool rotary_interleaved, const int64_t num_splits,
           const bool return_softmax_lse, Tensor out,
           std::optional<Tensor> softmax_lse)
      : FlashAttnWithKvcache(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
                             cache_seqlens, cache_batch_idx, cache_leftpad,
                             block_table, alibi_slopes, softmax_scale, causal,
                             window_size, softcap, rotary_interleaved,
                             num_splits, return_softmax_lse, out, softmax_lse) {
  }

  bool capture_safe() const override { return true; }

  std::size_t workspace_size_in_bytes() const override {
    if (q_shape_.size() < 3) return 0;
    return static_cast<std::size_t>(q_shape_[0]) *
           static_cast<std::size_t>(q_shape_[1]) *
           static_cast<std::size_t>(q_shape_[2]) * sizeof(float);
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const int64_t cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override;

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const std::optional<Tensor> cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MARS_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
