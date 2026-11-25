#ifndef INFINI_OPS_MARS_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_MARS_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <optional>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "native/cuda/mars/ops/flash_attn_common.h"

namespace infini::ops {

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kMars, 0>
    : public FlashAttnVarlenFunc {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
           const int64_t max_seqlen_k, const double dropout_p,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool deterministic, const bool return_attn_probs, Tensor out,
           std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask)
      : FlashAttnVarlenFunc(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                            block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                            softmax_scale, causal, window_size, softcap,
                            deterministic, return_attn_probs, out, softmax_lse,
                            s_dmask) {}

  bool capture_safe() const override { return true; }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MARS_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
