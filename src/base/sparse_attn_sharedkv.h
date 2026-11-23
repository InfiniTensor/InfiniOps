#ifndef INFINI_OPS_BASE_SPARSE_ATTN_SHAREDKV_H_
#define INFINI_OPS_BASE_SPARSE_ATTN_SHAREDKV_H_
#include <cstdint>
#include <string>
#include "operator.h"
#include "tensor.h"
namespace infini::ops {
class SparseAttnSharedkv : public Operator<SparseAttnSharedkv> {
 public:
  static constexpr bool kAutomaticDefault = true;
  SparseAttnSharedkv(Tensor output, Tensor softmax_lse, const Tensor q,
      const Tensor ori_kv, const Tensor cmp_kv, const Tensor ori_sparse_indices,
      const Tensor cmp_sparse_indices, const Tensor ori_block_table,
      const Tensor cmp_block_table, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv, const Tensor sinks,
      const Tensor metadata, double softmax_scale, int64_t cmp_ratio,
      int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_kv_stride,
      int64_t cmp_kv_stride, int64_t ori_win_left, int64_t ori_win_right,
      std::string layout_q, std::string layout_kv, bool return_softmax_lse) {}
  virtual void operator()(Tensor output, Tensor softmax_lse, const Tensor q,
      const Tensor ori_kv, const Tensor cmp_kv, const Tensor ori_sparse_indices,
      const Tensor cmp_sparse_indices, const Tensor ori_block_table,
      const Tensor cmp_block_table, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_ori_kv, const Tensor cu_seqlens_cmp_kv,
      const Tensor seqused_q, const Tensor seqused_kv, const Tensor sinks,
      const Tensor metadata, double softmax_scale, int64_t cmp_ratio,
      int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_kv_stride,
      int64_t cmp_kv_stride, int64_t ori_win_left, int64_t ori_win_right,
      std::string layout_q, std::string layout_kv, bool return_softmax_lse) const = 0;
};
}
#endif
