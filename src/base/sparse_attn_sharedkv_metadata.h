#ifndef INFINI_OPS_BASE_SPARSE_ATTN_SHAREDKV_METADATA_H_
#define INFINI_OPS_BASE_SPARSE_ATTN_SHAREDKV_METADATA_H_
#include <cstdint>
#include <string>
#include "operator.h"
#include "tensor.h"
namespace infini::ops {
class SparseAttnSharedkvMetadata : public Operator<SparseAttnSharedkvMetadata> {
 public:
  static constexpr bool kAutomaticDefault = true;
  SparseAttnSharedkvMetadata(Tensor output, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv,
      int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
      int64_t block_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
      int64_t ori_topk, int64_t cmp_topk, int64_t cmp_ratio,
      int64_t ori_mask_mode, int64_t cmp_mask_mode,
      int64_t ori_window_left, int64_t ori_window_right,
      std::string layout_q, std::string layout_kv,
      bool has_ori_sparse, bool has_cmp_sparse) {}
  virtual void operator()(Tensor output, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv,
      int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
      int64_t block_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
      int64_t ori_topk, int64_t cmp_topk, int64_t cmp_ratio,
      int64_t ori_mask_mode, int64_t cmp_mask_mode,
      int64_t ori_window_left, int64_t ori_window_right,
      std::string layout_q, std::string layout_kv,
      bool has_ori_sparse, bool has_cmp_sparse) const = 0;
};
}
#endif
