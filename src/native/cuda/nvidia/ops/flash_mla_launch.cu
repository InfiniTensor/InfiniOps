#include <cuda_runtime.h>

#include "dispatcher.h"
#include "native/cuda/nvidia/data_type_.h"
#include "native/cuda/nvidia/ops/flash_mla_launch.h"
#include "native/cuda/ops/flash_mla_warp.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops::native_flash_mla {

void launch_mla_prefill(void* out, const void* q, const void* k_cache,
                        const void* v_cache, const void* block_tables,
                        const void* cu_seqlens_k, const void* cu_seqlens_q,
                        DataType data_dtype, DataType index_dtype,
                        const MlaPrefillLaunch& p, cudaStream_t stream) {
  DispatchFunc<ReducedFloatTypes, List<DataType::kInt32, DataType::kInt64>,
               List<576>>(
      {static_cast<int64_t>(data_dtype), static_cast<int64_t>(index_dtype),
       static_cast<int64_t>(576)},
      [&](auto list_tag) {
        using TData = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using TIndex = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;
        constexpr int kHeadSize = ListGet<2>(list_tag);
        constexpr int kValueHeadSize = 512;
        dim3 grid(static_cast<unsigned>(p.num_heads),
                  static_cast<unsigned>(p.total_qtokens));
        op::flash::cuda::FlashMlaPrefillWarpKernel<TIndex, TData, kHeadSize,
                                                   kValueHeadSize>
            <<<grid, 32, 0, stream>>>(
                static_cast<TData*>(out), static_cast<const TData*>(q),
                static_cast<const TData*>(k_cache),
                static_cast<const TData*>(v_cache),
                static_cast<const TIndex*>(block_tables),
                static_cast<const TIndex*>(cu_seqlens_k),
                static_cast<const TIndex*>(cu_seqlens_q), p.num_heads,
                p.num_seqs, p.num_kv_heads, p.total_qtokens, p.scale,
                p.max_num_blocks_per_seq, p.block_size,
                p.block_table_batch_stride, p.q_stride, p.q_head_stride,
                p.k_batch_stride, p.k_row_stride, p.k_head_stride,
                p.v_batch_stride, p.v_row_stride, p.v_head_stride, p.o_stride,
                p.o_head_stride);
      },
      "native_flash_mla::launch_mla_prefill");
}

void launch_mla_decode(void* out, const void* q, const void* k_cache,
                       const void* v_cache, const void* block_tables,
                       const void* seqlens, DataType data_dtype,
                       DataType index_dtype, const MlaDecodeLaunch& p,
                       cudaStream_t stream) {
  DispatchFunc<ReducedFloatTypes, List<DataType::kInt32, DataType::kInt64>,
               List<576>>(
      {static_cast<int64_t>(data_dtype), static_cast<int64_t>(index_dtype),
       static_cast<int64_t>(576)},
      [&](auto list_tag) {
        using TData = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using TIndex = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;
        constexpr int kHeadSize = ListGet<2>(list_tag);
        constexpr int kValueHeadSize = 512;
        dim3 grid(static_cast<unsigned>(p.num_heads),
                  static_cast<unsigned>(p.num_seqs));
        op::flash::cuda::FlashMlaDecodeWarpKernel<TIndex, TData, kHeadSize,
                                                  kValueHeadSize>
            <<<grid, 32, 0, stream>>>(
                static_cast<TData*>(out), static_cast<const TData*>(q),
                static_cast<const TData*>(k_cache),
                static_cast<const TData*>(v_cache),
                static_cast<const TIndex*>(block_tables),
                static_cast<const TIndex*>(seqlens), p.num_heads,
                p.num_kv_heads, p.scale, p.max_num_blocks_per_seq, p.block_size,
                p.q_batch_stride, p.q_head_stride, p.k_block_stride,
                p.k_head_stride, p.k_slot_stride, p.v_block_stride,
                p.v_head_stride, p.v_slot_stride, p.o_batch_stride,
                p.o_head_stride, p.block_table_batch_stride, p.seqlens_stride);
      },
      "native_flash_mla::launch_mla_decode");
}

}  // namespace infini::ops::native_flash_mla
