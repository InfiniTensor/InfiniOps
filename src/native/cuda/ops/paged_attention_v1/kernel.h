#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_H_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "base/paged_attention_v1.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/paged_attention_v1/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

using PagedAttentionV1IndexTypes =
    List<DataType::kInt32, DataType::kInt64, DataType::kUInt32>;

template <typename Backend>
class CudaPagedAttentionV1 : public PagedAttentionV1 {
 public:
  using PagedAttentionV1::PagedAttentionV1;

  void operator()(const Tensor query, const Tensor key_cache,
                  const Tensor value_cache, const Tensor block_tables,
                  const Tensor seq_lens,
                  const std::optional<Tensor> alibi_slopes,
                  const int64_t num_kv_heads, const double scale,
                  const int64_t block_size, const int64_t max_seq_len,
                  const std::string kv_cache_dtype, const double k_scale,
                  const double v_scale, const int64_t tp_rank,
                  const int64_t blocksparse_local_blocks,
                  const int64_t blocksparse_vert_stride,
                  const int64_t blocksparse_block_size,
                  const int64_t blocksparse_head_sliding_step,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    assert(query.dtype() == dtype_ && key_cache.dtype() == dtype_ &&
           value_cache.dtype() == dtype_ && out.dtype() == dtype_);
    assert(block_tables.dtype() == index_dtype_ &&
           seq_lens.dtype() == index_dtype_);
    assert(num_kv_heads == static_cast<int64_t>(num_kv_heads_));
    assert(scale == scale_);
    assert(block_size == static_cast<int64_t>(block_size_));
    assert(max_seq_len > 0);
    assert(kv_cache_dtype == "auto" && k_scale == 1.0 && v_scale == 1.0);
    assert(blocksparse_local_blocks == 0 && blocksparse_vert_stride == 0 &&
           blocksparse_head_sliding_step == 0);
    assert(blocksparse_block_size > 0);

    (void)tp_rank;

    dim3 grid(static_cast<unsigned>(num_heads_),
              static_cast<unsigned>(num_seqs_));

    DispatchFunc<ReducedFloatTypes, PagedAttentionV1IndexTypes, List<64, 128>>(
        {static_cast<int64_t>(dtype_), static_cast<int64_t>(index_dtype_),
         static_cast<int64_t>(head_size_)},
        [&](auto list_tag) {
          using TData = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TIndex =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kHeadSize = ListGet<2>(list_tag);

          PagedAttentionDecodeWarpKernel<TIndex, TData, kHeadSize>
              <<<grid, 32, 0, cuda_stream>>>(
                  reinterpret_cast<TData*>(out.data()),
                  reinterpret_cast<const TData*>(query.data()),
                  reinterpret_cast<const TData*>(key_cache.data()),
                  reinterpret_cast<const TData*>(value_cache.data()),
                  reinterpret_cast<const TIndex*>(block_tables.data()),
                  reinterpret_cast<const TIndex*>(seq_lens.data()),
                  alibi_slopes.has_value()
                      ? reinterpret_cast<const float*>(alibi_slopes->data())
                      : nullptr,
                  num_heads_, num_kv_heads_, static_cast<float>(scale),
                  max_num_blocks_per_seq_, block_size_, key_cache_block_stride_,
                  key_cache_head_stride_, key_cache_slot_stride_,
                  key_cache_dim_stride_, key_cache_x_stride_,
                  static_cast<int>(key_cache_x_), value_cache_block_stride_,
                  value_cache_head_stride_, value_cache_slot_stride_,
                  value_cache_dim_stride_, query_stride_, query_head_stride_,
                  out_stride_, out_head_stride_, block_table_batch_stride_,
                  seq_lens_stride_);
        },
        "CudaPagedAttentionV1::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_PAGED_ATTENTION_V1_KERNEL_H_
