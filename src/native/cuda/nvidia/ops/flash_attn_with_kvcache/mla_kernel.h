#ifndef INFINI_OPS_NVIDIA_FLASH_ATTN_WITH_KVCACHE_MLA_KERNEL_H_
#define INFINI_OPS_NVIDIA_FLASH_ATTN_WITH_KVCACHE_MLA_KERNEL_H_

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/flash_attn_with_kvcache.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/runtime_utils.h"

#define INFINI_NATIVE_KVCACHE_MLA_PARAMS(Lengths)                          \
  const Tensor q, Tensor k_cache, Tensor v_cache,                          \
      const std::optional<Tensor> k, const std::optional<Tensor> v,        \
      const std::optional<Tensor> rotary_cos,                              \
      const std::optional<Tensor> rotary_sin, const Lengths cache_seqlens, \
      const std::optional<Tensor> cache_batch_idx,                         \
      const std::optional<Tensor> cache_leftpad,                           \
      const std::optional<Tensor> block_table,                             \
      const std::optional<Tensor> alibi_slopes,                            \
      const std::optional<double> softmax_scale, const bool causal,        \
      const std::vector<int64_t> window_size, const double softcap,        \
      const bool rotary_interleaved, const int64_t num_splits,             \
      const bool return_softmax_lse, Tensor out,                           \
      std::optional<Tensor> softmax_lse
#define INFINI_NATIVE_KVCACHE_MLA_VALUES                               \
  q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,    \
      cache_batch_idx, cache_leftpad, block_table, alibi_slopes,       \
      softmax_scale, causal, window_size, softcap, rotary_interleaved, \
      num_splits, return_softmax_lse, out, softmax_lse

namespace infini::ops {

// MLA-packed decode over the fused latent KV cache: q/k head dim 576
// (kv_lora_rank 512 + rope 64) with a separate 512-dim value latent. This
// provider backs the base interface where the pinned FlashAttention kernels
// stop at head dim 256; all other shapes stay with the pinned provider.
template <>
class Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 1>
    : public FlashAttnWithKvcache {
 public:
  static bool is_mla_layout(Tensor k_cache, Tensor v_cache) {
    return k_cache.shape() != v_cache.shape();
  }

  template <typename Lengths>
  static bool supports(INFINI_NATIVE_KVCACHE_MLA_PARAMS(Lengths)) {
    if constexpr (!std::is_same_v<Lengths, std::optional<Tensor>>) {
      return false;  // Scalar metadata needs a separate prepared device buffer.
    } else {
      if (!FlashAttnWithKvcache::supports(INFINI_NATIVE_KVCACHE_MLA_VALUES))
        return false;
      if (!is_mla_layout(k_cache, v_cache)) return false;
      if (k || v || rotary_cos || rotary_sin || cache_batch_idx ||
          cache_leftpad || alibi_slopes)
        return false;
      if (softcap != 0.0 || num_splits != 0 || return_softmax_lse) return false;
      if (!(window_size[0] == -1 && window_size[1] == -1) &&
          !(window_size[0] >= 0 && window_size[1] == 0))
        return false;
      if (!causal) return false;
      if (!block_table || q.size(1) != 1) return false;
      if (q.size(3) != 576 || v_cache.size(3) != 512) return false;
      if (k_cache.size(1) != v_cache.size(1) ||
          k_cache.size(2) != v_cache.size(2))
        return false;
      // The fused MLA cache packs k|v in one 1088-wide row; K and V are
      // narrow views of it, so token strides are shared and only the head
      // dims differ. The kernel reads each cache through its own strides.
      if (k_cache.stride(3) != 1 || v_cache.stride(3) != 1 ||
          k_cache.stride(2) != v_cache.stride(2) ||
          k_cache.stride(1) != v_cache.stride(1) ||
          k_cache.stride(0) != v_cache.stride(0))
        return false;
      if (q.stride(3) != 1 || out.stride(3) != 1) return false;
      if (q.stride(0) != q.size(1) * q.size(2) * q.size(3) ||
          out.stride(0) != out.size(1) * out.size(2) * out.size(3))
        return false;
      // Dim 1 is the singleton token axis (enforced above), and unsqueezed
      // views carry a meaningless zero stride there, so it is not checked.
      if (q.stride(2) != q.size(3) || out.stride(2) != out.size(3))
        return false;
      if (q.stride(2) != q.size(3) || out.stride(2) != out.size(3))
        return false;
      return true;
    }
  }

  template <typename Lengths>
  Operator(INFINI_NATIVE_KVCACHE_MLA_PARAMS(Lengths))
      : FlashAttnWithKvcache{checked_q(INFINI_NATIVE_KVCACHE_MLA_VALUES),
                             k_cache,
                             v_cache,
                             k,
                             v,
                             rotary_cos,
                             rotary_sin,
                             cache_seqlens,
                             cache_batch_idx,
                             cache_leftpad,
                             block_table,
                             alibi_slopes,
                             softmax_scale,
                             causal,
                             window_size,
                             softcap,
                             rotary_interleaved,
                             num_splits,
                             return_softmax_lse,
                             out,
                             softmax_lse} {
    if constexpr (std::is_same_v<Lengths, std::optional<Tensor>>) {
      prepare(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale,
              window_size);
    }
  }
  ~Operator() override;
  using FlashAttnWithKvcache::operator();

  void operator()(INFINI_NATIVE_KVCACHE_MLA_PARAMS(int64_t)) const override {
    throw std::invalid_argument(
        "native MLA KV Attention requires device length metadata");
  }
  void operator()(
      INFINI_NATIVE_KVCACHE_MLA_PARAMS(std::optional<Tensor>)) const override {
    checked_q(INFINI_NATIVE_KVCACHE_MLA_VALUES);
    execute(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale,
            window_size, out);
  }

  // Graph callers must supply stable external workspace for the graph lifetime.
  bool capture_safe() const override { return true; }
  std::size_t workspace_size_in_bytes() const override { return 0; }

 private:
  template <typename Lengths>
  static Tensor checked_q(INFINI_NATIVE_KVCACHE_MLA_PARAMS(Lengths)) {
    if (!supports(INFINI_NATIVE_KVCACHE_MLA_VALUES)) {
      throw std::invalid_argument(
          "unsupported native MLA KV Attention configuration");
    }
    return q;
  }
  struct Impl;
  std::shared_ptr<Impl> impl_;
  void prepare(const Tensor& q, Tensor k_cache, Tensor v_cache,
               const std::optional<Tensor>& lengths,
               const std::optional<Tensor>& blocks,
               const std::optional<double>& scale,
               const std::vector<int64_t>& window_size);
  void execute(const Tensor& q, Tensor k_cache, Tensor v_cache,
               const std::optional<Tensor>& lengths,
               const std::optional<Tensor>& blocks,
               const std::optional<double>& scale,
               const std::vector<int64_t>& window_size, Tensor out) const;
};

}  // namespace infini::ops

#undef INFINI_NATIVE_KVCACHE_MLA_PARAMS
#undef INFINI_NATIVE_KVCACHE_MLA_VALUES

#endif
