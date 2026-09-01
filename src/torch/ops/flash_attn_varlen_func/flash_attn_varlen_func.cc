#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/equal.h>

#include <cassert>
#include <cmath>
#include <tuple>

#if defined(WITH_NVIDIA)
#include "torch/nvidia/c10.h"
#endif
#if defined(WITH_MOORE)
#include "native/cuda/moore/ops/flash_attn_varlen_func/paged.h"
#include "torch/moore/c10.h"
#endif
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void AtenFlashAttnVarlenFunc<kDev>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  assert(!alibi_slopes.has_value() &&
         "The PyTorch `FlashAttnVarlenFunc` provider does not support "
         "`alibi_slopes`.");
  assert(!block_table.has_value() &&
         "The PyTorch `FlashAttnVarlenFunc` provider does not support "
         "`block_table`.");
  if constexpr (kDev == Device::Type::kMoore) {
    assert(window_size[0] == -1 && window_size[1] == -1 &&
           "TorchMusa FlashAttention does not support local windows");
  }

  (void)softcap;
  (void)deterministic;
  (void)return_attn_probs;

  const auto run = [&]() {
    auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                   q_strides_, q_dtype_, device_index_);
    auto at_k = ToAtenTensor<kDev>(const_cast<void*>(k.data()), k_shape_,
                                   k_strides_, k_dtype_, device_index_);
    auto at_v = ToAtenTensor<kDev>(const_cast<void*>(v.data()), v_shape_,
                                   v_strides_, v_dtype_, device_index_);
    auto at_cu_seqlens_q = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                     out_dtype_, device_index_);
    std::optional<at::Tensor> at_softmax_lse;
    std::optional<at::Tensor> at_s_dmask;
    if (softmax_lse.has_value()) {
      at_softmax_lse.emplace(ToAtenTensor<kDev>(
          softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
          softmax_lse_dtype_, device_index_));
      at_s_dmask.emplace(ToAtenTensor<kDev>(s_dmask->data(), s_dmask_shape_,
                                            s_dmask_strides_, s_dmask_dtype_,
                                            device_index_));
    }

    if constexpr (kDev == Device::Type::kMoore) {
      assert((!causal || at::equal(at_cu_seqlens_q, at_cu_seqlens_k)) &&
             "TorchMusa causal FlashAttention requires matching query and "
             "key sequence lengths");
    }

    std::optional<int64_t> window_size_left;
    std::optional<int64_t> window_size_right;
    if constexpr (kDev == Device::Type::kNvidia) {
      if (window_size[0] >= 0) {
        window_size_left = window_size[0];
      }
      if (causal) {
        window_size_right = 0;
      } else if (window_size[1] >= 0) {
        window_size_right = window_size[1];
      }
    }

    auto result = at::_flash_attention_forward(
        at_q, at_k, at_v, at_cu_seqlens_q, at_cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, causal, false, softmax_scale, window_size_left,
        window_size_right, std::nullopt, std::nullopt);

    // ATen owns the returned tensors. Preserve the InfiniOps trailing-output
    // ABI by copying them into caller-provided buffers on the selected stream.
    at_out.copy_(std::get<0>(result));
    if (at_softmax_lse.has_value()) {
      const auto& result_softmax_lse = std::get<1>(result);
      if (result_softmax_lse.dim() == 3) {
        const auto batch_size = result_softmax_lse.size(0);
        const auto q_lengths = at_cu_seqlens_q.narrow(0, 1, batch_size) -
                               at_cu_seqlens_q.narrow(0, 0, batch_size);
        const auto positions =
            at::arange(result_softmax_lse.size(2), at_cu_seqlens_q.options());
        const auto valid_positions =
            positions.unsqueeze(0).lt(q_lengths.unsqueeze(1));
        const auto packed_softmax_lse =
            result_softmax_lse.transpose(1, 2)
                .masked_select(valid_positions.unsqueeze(2))
                .view({at_q.size(0), at_q.size(1)})
                .transpose(0, 1);
        at_softmax_lse->copy_(packed_softmax_lse);
      } else {
        at_softmax_lse->copy_(result_softmax_lse);
      }
      at_s_dmask->copy_(std::get<4>(result));
    }
  };

  using Backend = C10<kDev>;
  const typename Backend::StreamGuard stream_guard{
      Backend::GetStreamFromExternal(stream_, device_index_)};
  run();
}

#if defined(WITH_NVIDIA)
template class AtenFlashAttnVarlenFunc<Device::Type::kNvidia>;
#endif
#if defined(WITH_MOORE)
template class AtenFlashAttnVarlenFunc<Device::Type::kMoore>;

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 8>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  if (!block_table.has_value()) {
    return Base::operator()(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                            block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                            softmax_scale, causal, window_size, softcap,
                            deterministic, return_attn_probs, out, softmax_lse,
                            s_dmask);
  }

  assert(causal && "Moore paged FlashAttnVarlenFunc requires causal attention");
  assert(window_size[0] == -1 && window_size[1] == -1 &&
         "Moore paged FlashAttnVarlenFunc does not support local windows");
  assert(!return_attn_probs && !softmax_lse.has_value() &&
         !s_dmask.has_value() &&
         "Moore paged FlashAttnVarlenFunc does not return attention "
         "probabilities");
  assert((q_shape_[2] == 64 || q_shape_[2] == 128) &&
         "Moore paged FlashAttnVarlenFunc supports head sizes 64 and 128");
  assert(max_seqlen_k <=
             static_cast<int64_t>(block_table_shape_[1] * k_shape_[1]) &&
         "Moore paged FlashAttnVarlenFunc maximum K length exceeds the "
         "block-table capacity");
  assert((!alibi_slopes.has_value() || alibi_slopes_shape_.size() == 1) &&
         "Moore paged FlashAttnVarlenFunc supports only one-dimensional "
         "ALiBi slopes");

  (void)dropout_p;
  (void)softcap;
  (void)deterministic;

  MoorePagedFlashAttnVarlenParams params{};
  params.out = out.data();
  params.q = q.data();
  params.k_cache = k.data();
  params.v_cache = v.data();
  params.block_table = static_cast<const int32_t*>(block_table->data());
  params.cu_seqlens_q = static_cast<const int32_t*>(cu_seqlens_q.data());
  params.cu_seqlens_k = static_cast<const int32_t*>(cu_seqlens_k.data());
  params.alibi_slopes = alibi_slopes.has_value()
                            ? static_cast<const float*>(alibi_slopes->data())
                            : nullptr;
  params.dtype = q_dtype_;
  params.total_q_tokens = q_shape_[0];
  params.num_heads = q_shape_[1];
  params.num_kv_heads = k_shape_[2];
  params.num_seqs = block_table_shape_[0];
  params.head_size = q_shape_[2];
  params.page_size = k_shape_[1];
  params.max_num_blocks_per_seq = block_table_shape_[1];
  params.max_seqlen_q = max_seqlen_q;
  params.block_table_batch_stride = block_table_strides_[0];
  params.q_stride = q_strides_[0];
  params.q_head_stride = q_strides_[1];
  params.k_cache_block_stride = k_strides_[0];
  params.k_cache_slot_stride = k_strides_[1];
  params.k_cache_head_stride = k_strides_[2];
  params.v_cache_block_stride = v_strides_[0];
  params.v_cache_slot_stride = v_strides_[1];
  params.v_cache_head_stride = v_strides_[2];
  params.out_stride = out_strides_[0];
  params.out_head_stride = out_strides_[1];
  params.scale = static_cast<float>(softmax_scale.value_or(
      1.0 / std::sqrt(static_cast<double>(q_shape_[2]))));
  params.stream = stream_;

  using Backend = C10<Device::Type::kMoore>;
  const typename Backend::StreamGuard stream_guard{
      Backend::GetStreamFromExternal(stream_, device_index_)};

  LaunchMoorePagedFlashAttnVarlenFunc(params);
}
#endif

}  // namespace infini::ops
