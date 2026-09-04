#include <cassert>

#include "dispatcher.h"
#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/ops/flash_attn_varlen_func/paged.h"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.cuh"

namespace infini::ops {

void LaunchMoorePagedFlashAttnVarlenFunc(
    const MoorePagedFlashAttnVarlenParams& params) {
  using Backend = Runtime<Device::Type::kMoore>;
  using Index = int32_t;
  using Lengths =
      op::paged_attention_prefill::cuda::CumulativeSequenceLengths<Index>;

  if (params.total_q_tokens == 0) {
    return;
  }

  assert(params.out != nullptr && params.q != nullptr &&
         params.k_cache != nullptr && params.v_cache != nullptr);
  assert(params.block_table != nullptr && params.cu_seqlens_q != nullptr &&
         params.cu_seqlens_k != nullptr);

  const auto stream =
      static_cast<Backend::Stream>(params.stream ? params.stream : 0);
  const Lengths kv_lengths{params.cu_seqlens_k};

  DispatchFunc<ReducedFloatTypes, List<64, 128>>(
      {static_cast<int64_t>(params.dtype),
       static_cast<int64_t>(params.head_size)},
      [&](auto list_tag) {
        using TData = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
        constexpr int kHeadSize = ListGet<1>(list_tag);

        constexpr std::size_t kMaxGridDimension = 65535;
        const std::size_t grid_y = params.total_q_tokens < kMaxGridDimension
                                       ? params.total_q_tokens
                                       : kMaxGridDimension;
        const std::size_t grid_z = (params.total_q_tokens - 1) / grid_y + 1;
        assert(grid_z <= kMaxGridDimension && "packed Q is too large");
        const dim3 grid(static_cast<unsigned>(params.num_heads),
                        static_cast<unsigned>(grid_y),
                        static_cast<unsigned>(grid_z));
        op::paged_attention_prefill::cuda::
            PagedAttentionPrefillWarpGlobalKernel<Device::Type::kMoore, Index,
                                                  TData, kHeadSize, Lengths>
            <<<grid, 32, 0, stream>>>(
                reinterpret_cast<TData*>(params.out),
                reinterpret_cast<const TData*>(params.q),
                reinterpret_cast<const TData*>(params.k_cache),
                reinterpret_cast<const TData*>(params.v_cache),
                params.block_table, kv_lengths, params.cu_seqlens_q,
                params.alibi_slopes, params.num_heads, params.num_seqs,
                params.num_kv_heads, params.total_q_tokens, params.scale,
                params.max_num_blocks_per_seq, params.page_size,
                params.block_table_batch_stride, params.q_stride,
                params.q_head_stride, params.k_cache_block_stride,
                params.k_cache_slot_stride, params.k_cache_head_stride,
                params.v_cache_block_stride, params.v_cache_slot_stride,
                params.v_cache_head_stride, params.out_stride,
                params.out_head_stride);
      },
      "LaunchMoorePagedFlashAttnVarlenFunc");
}

}  // namespace infini::ops
