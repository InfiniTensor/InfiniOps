#ifndef INFINI_OPS_BASE_PAGED_ATTENTION_V1_H_
#define INFINI_OPS_BASE_PAGED_ATTENTION_V1_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// Aligned with vLLM `_custom_ops.paged_attention_v1`.
class PagedAttentionV1 : public Operator<PagedAttentionV1> {
 public:
  PagedAttentionV1(const Tensor query, const Tensor key_cache,
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
                   const int64_t blocksparse_head_sliding_step, Tensor out)
      : dtype_{query.dtype()},
        index_dtype_{block_tables.dtype()},
        num_seqs_{query.size(0)},
        num_heads_{query.size(1)},
        num_kv_heads_{key_cache.size(1)},
        head_size_{query.size(2)},
        block_size_{key_cache.size(3)},
        max_num_blocks_per_seq_{block_tables.size(1)},
        key_cache_x_{key_cache.size(4)},
        query_stride_{query.stride(0)},
        query_head_stride_{query.stride(1)},
        key_cache_block_stride_{key_cache.stride(0)},
        key_cache_head_stride_{key_cache.stride(1)},
        key_cache_dim_stride_{key_cache.stride(2)},
        key_cache_slot_stride_{key_cache.stride(3)},
        key_cache_x_stride_{key_cache.stride(4)},
        value_cache_block_stride_{value_cache.stride(0)},
        value_cache_head_stride_{value_cache.stride(1)},
        value_cache_dim_stride_{value_cache.stride(2)},
        value_cache_slot_stride_{value_cache.stride(3)},
        out_stride_{out.stride(0)},
        out_head_stride_{out.stride(1)},
        block_table_batch_stride_{block_tables.stride(0)},
        seq_lens_stride_{seq_lens.stride(0)},
        scale_{scale},
        device_index_{query.device().index()} {
    assert(query.ndim() == 3 && out.ndim() == 3 &&
           "`PagedAttentionV1` requires 3D `query` and `out`");
    assert(key_cache.ndim() == 5 && value_cache.ndim() == 4 &&
           "`PagedAttentionV1` requires a 5D key cache and 4D value cache");
    assert(block_tables.ndim() == 2 && seq_lens.ndim() == 1 &&
           "`PagedAttentionV1` requires 2D block tables and 1D sequence "
           "lengths");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "`PagedAttentionV1` supports float16 and bfloat16");
    assert(out.dtype() == dtype_ && key_cache.dtype() == dtype_ &&
           value_cache.dtype() == dtype_ &&
           "`PagedAttentionV1` requires matching data dtypes");
    assert(IsIndexDtype(index_dtype_) && seq_lens.dtype() == index_dtype_ &&
           "`PagedAttentionV1` requires matching integer metadata dtypes");
    assert(query.shape() == out.shape() &&
           "`PagedAttentionV1` requires `out` to match `query`");
    assert(num_kv_heads > 0 &&
           static_cast<Tensor::Size>(num_kv_heads) == num_kv_heads_ &&
           num_heads_ % num_kv_heads_ == 0 &&
           "`PagedAttentionV1` received incompatible KV heads");
    assert(key_cache.size(0) == value_cache.size(0) &&
           value_cache.size(1) == num_kv_heads_ &&
           key_cache.size(2) * key_cache_x_ == head_size_ &&
           value_cache.size(2) == head_size_ &&
           "`PagedAttentionV1` cache shapes do not match `query`");
    assert(block_size > 0 &&
           static_cast<Tensor::Size>(block_size) == block_size_ &&
           value_cache.size(3) == block_size_ &&
           "`PagedAttentionV1` received an incompatible block size");
    assert(key_cache_x_ ==
               static_cast<Tensor::Size>(16 / query.element_size()) &&
           "`PagedAttentionV1` requires the vLLM vectorized key layout");
    assert(max_seq_len > 0 &&
           max_seq_len <=
               static_cast<int64_t>(max_num_blocks_per_seq_ * block_size_) &&
           "`PagedAttentionV1` maximum sequence length exceeds block table "
           "capacity");
    assert((head_size_ == 64 || head_size_ == 128) &&
           "`PagedAttentionV1` supports head sizes 64 and 128");
    assert(block_tables.size(0) == num_seqs_ && seq_lens.size(0) == num_seqs_ &&
           "`PagedAttentionV1` metadata batch sizes must match `query`");
    assert(query.stride(2) == 1 && out.stride(2) == 1 &&
           key_cache_x_stride_ == 1 &&
           "`PagedAttentionV1` requires contiguous head-vector dimensions");
    assert(block_tables.stride(1) == 1 && seq_lens_stride_ == 1 &&
           "`PagedAttentionV1` requires contiguous index rows");
    assert(!alibi_slopes.has_value() ||
           (alibi_slopes->dtype() == DataType::kFloat32 &&
            alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == num_heads_ &&
            alibi_slopes->stride(0) == 1) &&
               "`PagedAttentionV1` received incompatible ALiBi slopes");
    assert(kv_cache_dtype == "auto" && k_scale == 1.0 && v_scale == 1.0 &&
           "`PagedAttentionV1` currently supports unquantized KV caches");
    assert(blocksparse_local_blocks == 0 && blocksparse_vert_stride == 0 &&
           blocksparse_head_sliding_step == 0 &&
           "`PagedAttentionV1` does not yet support block-sparse attention");
    assert(blocksparse_block_size > 0 &&
           "`PagedAttentionV1` requires a positive block-sparse block size");

    const auto same_device_as_query = [&](const Tensor tensor) {
      return tensor.device() == query.device();
    };
    assert(same_device_as_query(key_cache) &&
           same_device_as_query(value_cache) &&
           same_device_as_query(block_tables) &&
           same_device_as_query(seq_lens) && same_device_as_query(out) &&
           (!alibi_slopes.has_value() || same_device_as_query(*alibi_slopes)) &&
           "`PagedAttentionV1` tensors must be on the same device");

    (void)tp_rank;
  }

  virtual void operator()(
      const Tensor query, const Tensor key_cache, const Tensor value_cache,
      const Tensor block_tables, const Tensor seq_lens,
      const std::optional<Tensor> alibi_slopes, const int64_t num_kv_heads,
      const double scale, const int64_t block_size, const int64_t max_seq_len,
      const std::string kv_cache_dtype, const double k_scale,
      const double v_scale, const int64_t tp_rank,
      const int64_t blocksparse_local_blocks,
      const int64_t blocksparse_vert_stride,
      const int64_t blocksparse_block_size,
      const int64_t blocksparse_head_sliding_step, Tensor out) const = 0;

 protected:
  static bool IsIndexDtype(DataType dtype) {
    return dtype == DataType::kInt32 || dtype == DataType::kInt64 ||
           dtype == DataType::kUInt32;
  }

  DataType dtype_;

  DataType index_dtype_;

  Tensor::Size num_seqs_{0};

  Tensor::Size num_heads_{0};

  Tensor::Size num_kv_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size block_size_{0};

  Tensor::Size max_num_blocks_per_seq_{0};

  Tensor::Size key_cache_x_{0};

  Tensor::Stride query_stride_{0};

  Tensor::Stride query_head_stride_{0};

  Tensor::Stride key_cache_block_stride_{0};

  Tensor::Stride key_cache_head_stride_{0};

  Tensor::Stride key_cache_dim_stride_{0};

  Tensor::Stride key_cache_slot_stride_{0};

  Tensor::Stride key_cache_x_stride_{0};

  Tensor::Stride value_cache_block_stride_{0};

  Tensor::Stride value_cache_head_stride_{0};

  Tensor::Stride value_cache_dim_stride_{0};

  Tensor::Stride value_cache_slot_stride_{0};

  Tensor::Stride out_stride_{0};

  Tensor::Stride out_head_stride_{0};

  Tensor::Stride block_table_batch_stride_{0};

  Tensor::Stride seq_lens_stride_{0};

  double scale_{1.0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_PAGED_ATTENTION_V1_H_
