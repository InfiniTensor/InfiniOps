#ifndef INFINI_OPS_NVIDIA_FLASH_ATTN_VARLEN_FUNC_MLA_KERNEL_H_
#define INFINI_OPS_NVIDIA_FLASH_ATTN_VARLEN_FUNC_MLA_KERNEL_H_

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/runtime_utils.h"

#define INFINI_NATIVE_VARLEN_MLA_ARGS                                        \
  const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q, \
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,   \
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,   \
      const int64_t max_seqlen_k, const double dropout_p,                    \
      const std::optional<double> softmax_scale, const bool causal,          \
      const std::vector<int64_t> window_size, const double softcap,          \
      const bool deterministic, const bool return_attn_probs, Tensor out,    \
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask
#define INFINI_NATIVE_VARLEN_MLA_VALUES                             \
  q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,   \
      max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, \
      window_size, softcap, deterministic, return_attn_probs, out,  \
      softmax_lse, s_dmask

namespace infini::ops {

// MLA-packed paged prefill over the fused latent KV cache: q/k head dim 576
// (kv_lora_rank 512 + rope 64) with a separate 512-dim value latent. This
// provider backs the base interface where the pinned FlashAttention kernels
// stop at head dim 256; all other shapes stay with the pinned provider.
template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 1>
    : public FlashAttnVarlenFunc {
 public:
  static bool is_mla_layout(const Tensor& k, const Tensor& v) {
    return k.shape() != v.shape();
  }

  static bool supports(INFINI_NATIVE_VARLEN_MLA_ARGS) {
    const bool mla = is_mla_layout(k, v);
    // Metadata gate first, then the MLA-specific shapes this kernel serves.
    if (!FlashAttnVarlenFunc::supports(INFINI_NATIVE_VARLEN_MLA_VALUES))
      return false;
    if (!mla) return false;  // Non-MLA belongs to the pinned provider.
    const bool paged = block_table.has_value();
    if (dropout_p != 0.0 || softcap != 0.0 || deterministic ||
        return_attn_probs || alibi_slopes)
      return false;
    if (!(window_size[0] == -1 && window_size[1] == -1) &&
        !(window_size[0] >= 0 && window_size[1] == 0))
      return false;
    // Causal prefill only; non-causal dense attention stays with the
    // pinned provider family.
    if (!causal) return false;
    if (!paged) return false;  // MLA prefill always reads the paged cache.
    if (q.size(2) != 576 || v.size(-1) != 512) return false;
    if (k.size(1) != v.size(1) || k.size(2) != v.size(2)) return false;
    // The fused MLA cache packs k|v in one 1088-wide row; K and V are
    // narrow views of it, so token strides are shared and only the head
    // dims differ. The kernel reads each cache through its own strides.
    if (k.stride(3) != 1 || v.stride(3) != 1 || k.stride(1) != v.stride(1) ||
        k.stride(2) != v.stride(2) || k.stride(0) != v.stride(0))
      return false;
    if (q.stride(2) != 1 || out.stride(2) != 1) return false;
    if (q.stride(0) != q.size(1) * q.size(2) ||
        out.stride(0) != out.size(1) * out.size(2))
      return false;
    if (q.stride(1) != q.size(2) || out.stride(1) != out.size(2)) return false;
    if (cu_seqlens_q.size(0) != cu_seqlens_k.size(0)) return false;
    return true;
  }

  Operator(INFINI_NATIVE_VARLEN_MLA_ARGS)
      : FlashAttnVarlenFunc(
            checked_q(INFINI_NATIVE_VARLEN_MLA_VALUES), k, v, cu_seqlens_q,
            cu_seqlens_k, alibi_slopes, block_table, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal, window_size, softcap,
            deterministic, return_attn_probs, out, softmax_lse, s_dmask) {
    prepare(INFINI_NATIVE_VARLEN_MLA_VALUES);
  }
  ~Operator() override;
  using FlashAttnVarlenFunc::operator();

  void operator()(INFINI_NATIVE_VARLEN_MLA_ARGS) const override;

  bool capture_safe() const override { return true; }
  std::size_t workspace_size_in_bytes() const override { return 0; }

 private:
  static Tensor checked_q(INFINI_NATIVE_VARLEN_MLA_ARGS) {
    if (!supports(INFINI_NATIVE_VARLEN_MLA_VALUES))
      throw std::invalid_argument(
          "unsupported native MLA varlen Attention configuration");
    return q;
  }
  struct Impl;
  std::shared_ptr<Impl> impl_;
  void prepare(INFINI_NATIVE_VARLEN_MLA_ARGS);
};

}  // namespace infini::ops

#undef INFINI_NATIVE_VARLEN_MLA_ARGS
#undef INFINI_NATIVE_VARLEN_MLA_VALUES

#endif
