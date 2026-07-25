#include "torch/moore/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/equal.h>
#include <c10/util/Exception.h>

#include "torch/tensor_.h"

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 8>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const std::optional<Tensor> alibi_slopes, const bool deterministic,
    const bool return_attn_probs, const std::optional<Tensor> block_table,
    Tensor out) const {
  TORCH_CHECK(window_size[0] < 0 && window_size[1] < 0,
              "TorchMusa FlashAttention does not support local windows");

  if (causal) {
    auto at_cu_seqlens_q = ToAtenTensor<Device::Type::kMoore>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<Device::Type::kMoore>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    TORCH_CHECK(at::equal(at_cu_seqlens_q, at_cu_seqlens_k),
                "TorchMusa FlashAttention requires matching query and key "
                "sequence lengths for causal attention");
  }

  AtenFlashAttnVarlenFunc<Device::Type::kMoore>::operator()(
      q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
      deterministic, return_attn_probs, block_table, out);
}

}  // namespace infini::ops
