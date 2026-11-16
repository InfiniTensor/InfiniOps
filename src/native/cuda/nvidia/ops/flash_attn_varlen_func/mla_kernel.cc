#include "mla_kernel.h"

#include <cuda_runtime.h>

#include <cmath>

#include "native/cuda/nvidia/ops/flash_mla_launch.h"
#include "native/cuda/nvidia/runtime_.h"

// The provider header undefines its macro; the definitions below re-spell
// the same public overloads locally, matching the pinned provider pattern.
#define MLA_VARLEN_ARGS                                                      \
  const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q, \
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,   \
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,   \
      const int64_t max_seqlen_k, const double dropout_p,                    \
      const std::optional<double> softmax_scale, const bool causal,          \
      const std::vector<int64_t> window_size, const double softcap,          \
      const bool deterministic, const bool return_attn_probs, Tensor out,    \
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask

namespace infini::ops {

struct Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 1>::Impl {
  native_flash_mla::MlaPrefillLaunch params{};
  std::optional<double> scale;
  bool requested_causal;
  std::array<int64_t, 2> requested_window;
};

void Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 1>::prepare(
    MLA_VARLEN_ARGS) {
  (void)alibi_slopes;
  (void)cu_seqlens_k;
  (void)max_seqlen_q;
  (void)max_seqlen_k;
  (void)dropout_p;
  (void)softcap;
  (void)deterministic;
  (void)return_attn_probs;
  (void)softmax_lse;
  (void)s_dmask;
  auto state = std::make_shared<Impl>();
  state->scale = softmax_scale;
  state->requested_causal = causal;
  state->requested_window = {window_size[0], window_size[1]};
  auto& p = state->params;
  p.num_heads = q.size(1);
  p.num_seqs = cu_seqlens_q.size(0) - 1;
  p.num_kv_heads = k.size(-2);
  p.total_qtokens = q.size(0);
  p.max_num_blocks_per_seq = block_table->size(1);
  p.block_size = k.size(1);
  p.scale = static_cast<float>(
      softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(2)))));
  if (!std::isfinite(p.scale))
    throw std::invalid_argument("nonfinite native MLA varlen Attention scale");
  p.block_table_batch_stride = block_table->stride(0);
  p.q_stride = q.stride(0);
  p.q_head_stride = q.stride(1);
  p.k_batch_stride = k.stride(0);
  p.k_row_stride = k.stride(1);
  p.k_head_stride = k.stride(2);
  p.v_batch_stride = v.stride(0);
  p.v_row_stride = v.stride(1);
  p.v_head_stride = v.stride(2);
  p.o_stride = out.stride(0);
  p.o_head_stride = out.stride(1);
  impl_ = std::move(state);
}

void Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 1>::operator()(
    MLA_VARLEN_ARGS) const {
  if (q.shape() != q_shape_ || k.shape() != k_shape_ || v.shape() != v_shape_ ||
      out.shape() != out_shape_ ||
      (block_table ? Tensor::Shape{block_table->shape()} : Tensor::Shape{}) !=
          block_table_shape_ ||
      softmax_scale != impl_->scale || causal != impl_->requested_causal ||
      window_size[0] != impl_->requested_window[0] ||
      window_size[1] != impl_->requested_window[1])
    throw std::invalid_argument(
        "native MLA varlen Attention prepared metadata changed");
  auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
  native_flash_mla::launch_mla_prefill(
      out.data(), q.data(), k.data(), v.data(), block_table->data(),
      cu_seqlens_k.data(), cu_seqlens_q.data(), q_dtype_, block_table_dtype_,
      impl_->params, stream);
}

Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 1>::~Operator() {
  impl_.reset();
}

}  // namespace infini::ops

#undef MLA_VARLEN_ARGS
