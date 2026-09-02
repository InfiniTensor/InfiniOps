#include <cassert>
#include <cmath>

#include "dispatcher.h"
#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/ops/flash_attn_with_kvcache/paged.h"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.cuh"

namespace infini::ops {

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 8>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  assert(false &&
         "Moore paged FlashAttnWithKvcache requires tensor cache_seqlens");
  (void)q;
  (void)k_cache;
  (void)v_cache;
  (void)k;
  (void)v;
  (void)rotary_cos;
  (void)rotary_sin;
  (void)cache_seqlens;
  (void)cache_batch_idx;
  (void)cache_leftpad;
  (void)block_table;
  (void)alibi_slopes;
  (void)softmax_scale;
  (void)causal;
  (void)window_size;
  (void)softcap;
  (void)rotary_interleaved;
  (void)num_splits;
  (void)return_softmax_lse;
  (void)out;
  (void)softmax_lse;
}

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 8>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
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
    std::optional<Tensor> softmax_lse) const {
  assert(cache_seqlens.has_value() && block_table.has_value() &&
         "Moore FlashAttnWithKvcache requires paged tensor metadata");
  assert(!k.has_value() && !v.has_value() && !rotary_cos.has_value() &&
         !rotary_sin.has_value() && !cache_batch_idx.has_value() &&
         !cache_leftpad.has_value() &&
         "Moore FlashAttnWithKvcache supports read-only paged decode");
  assert(q_shape_[1] == 1 &&
         "Moore FlashAttnWithKvcache supports one decode token per batch");
  assert(causal && window_size[0] == -1 && window_size[1] == -1 &&
         softcap == 0.0 && num_splits == 0 &&
         "Moore FlashAttnWithKvcache supports global causal attention");
  assert(!return_softmax_lse && !softmax_lse.has_value() &&
         "Moore FlashAttnWithKvcache does not return softmax LSE");
  assert((head_size_ == 64 || head_size_ == 128) &&
         "Moore FlashAttnWithKvcache supports head sizes 64 and 128");
  assert((!alibi_slopes.has_value() || alibi_slopes_shape_.size() == 1) &&
         "Moore FlashAttnWithKvcache supports one-dimensional ALiBi slopes");

  (void)rotary_interleaved;

  using Backend = Runtime<Device::Type::kMoore>;
  using Index = int32_t;
  const auto stream = static_cast<Backend::Stream>(stream_ ? stream_ : 0);
  const dim3 grid(static_cast<unsigned>(q_shape_[2]),
                  static_cast<unsigned>(q_shape_[0]));
  const float scale = static_cast<float>(
      softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_size_))));

  DispatchFunc<ReducedFloatTypes, List<64, 128>>(
      {static_cast<int64_t>(q_dtype_), static_cast<int64_t>(head_size_)},
      [&](auto list_tag) {
        using TData = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
        constexpr int kHeadSize = ListGet<1>(list_tag);

        PagedAttentionInfinilmDecodeWarpKernel<Index, TData, kHeadSize>
            <<<grid, 32, 0, stream>>>(
                reinterpret_cast<TData*>(out.data()),
                reinterpret_cast<const TData*>(q.data()),
                reinterpret_cast<const TData*>(k_cache.data()),
                reinterpret_cast<const TData*>(v_cache.data()),
                reinterpret_cast<const Index*>(block_table->data()),
                reinterpret_cast<const Index*>(cache_seqlens->data()),
                alibi_slopes.has_value()
                    ? reinterpret_cast<const float*>(alibi_slopes->data())
                    : nullptr,
                q_shape_[2], k_cache_shape_[2], scale, block_table_shape_[1],
                k_cache_shape_[1], k_cache_strides_[0], k_cache_strides_[2],
                k_cache_strides_[1], v_cache_strides_[0], v_cache_strides_[2],
                v_cache_strides_[1], q_strides_[0], q_strides_[2],
                out_strides_[0], out_strides_[2], block_table_strides_[0],
                cache_seqlens_strides_[0]);
      },
      "MooreFlashAttnWithKvcache");
}

}  // namespace infini::ops
