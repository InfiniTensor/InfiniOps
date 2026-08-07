#ifndef INFINI_OPS_ASCEND_MHA_FWD_KVCACHE_KERNEL_H_
#define INFINI_OPS_ASCEND_MHA_FWD_KVCACHE_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "base/mha_fwd_kvcache.h"
#include "native/ascend/common.h"
#include "native/ascend/ops/fia_common_.h"
#include "native/ascend/ops/graph_cleanup_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<MhaFwdKvcache, Device::Type::kAscend> : public MhaFwdKvcache {
 public:
  Operator(const Tensor q, const Tensor kcache, const Tensor vcache,
           std::optional<Tensor> k, std::optional<Tensor> v,
           std::optional<Tensor> seqlens_k, std::optional<Tensor> rotary_cos,
           std::optional<Tensor> rotary_sin,
           std::optional<Tensor> cache_batch_idx,
           std::optional<Tensor> leftpad_k, std::optional<Tensor> block_table,
           std::optional<Tensor> alibi_slopes, Tensor out, float softmax_scale,
           bool is_causal, int64_t window_size_left, int64_t window_size_right,
           float softcap, bool is_rotary_interleaved, int64_t num_splits = 0)
      : MhaFwdKvcache(q, kcache, vcache, k, v, seqlens_k, rotary_cos,
                      rotary_sin, cache_batch_idx, leftpad_k, block_table,
                      alibi_slopes, out, softmax_scale, is_causal,
                      window_size_left, window_size_right, softcap,
                      is_rotary_interleaved, num_splits) {
    ascend::fia::AssertSupportedDtype(q.dtype(), "`MhaFwdKvcache`");

    assert(seqlen_q_ == 1 &&
           "`MhaFwdKvcache`: this Ascend path supports decode "
           "`seqlen_q == 1` only");
    assert(has_block_table_ &&
           "`MhaFwdKvcache`: this Ascend path requires paged KV "
           "`block_table`");
    assert(has_seqlens_k_ &&
           "`MhaFwdKvcache`: this Ascend path requires `seqlens_k`");
    assert(page_block_size_ > 0 && page_block_size_ <= 512 &&
           page_block_size_ % 16 == 0 &&
           "`MhaFwdKvcache`: paged KV block size must be 16-aligned and "
           "not exceed 512 for `float16` and `bfloat16`");

    auto acl_dt = ascend::ToAclDtype(q.dtype());
    const auto B = static_cast<int64_t>(q.size(0));
    const auto N = static_cast<int64_t>(q.size(2));
    const auto D = static_cast<int64_t>(q.size(3));
    q_cache_ = ascend::AclTensorCache({B, N, 1, D}, acl_dt,
                                      const_cast<void*>(q.data()));
    out_cache_ = ascend::AclTensorCache({B, N, 1, D}, acl_dt, out.data());
    block_table_cache_ = ascend::AclTensorCache(block_table.value());

    const int64_t num_blocks = kcache.size(0);
    const int64_t block_size = kcache.size(1);
    const int64_t num_kv_heads = kcache.size(2);
    const int64_t head_dim = kcache.size(3);
    // FIA decode expects paged KV descriptors as `[num_blocks, num_kv_heads,
    // block_size, head_dim]`.  The physical cache is stored as
    // `[num_blocks, block_size, num_kv_heads, head_dim]`, so describe it as a
    // strided view instead of moving data.
    kv_shape_ = {num_blocks, num_kv_heads, block_size, head_dim};
    kv_strides_ = {block_size * num_kv_heads * head_dim, head_dim,
                   num_kv_heads * head_dim, 1};
    kv_storage_shape_ = {num_blocks * block_size * num_kv_heads * head_dim};
    kv_acl_dtype_ = acl_dt;
  }

  void operator()(
      const Tensor q, const Tensor kcache, const Tensor vcache,
      std::optional<Tensor> k, std::optional<Tensor> v,
      std::optional<Tensor> seqlens_k, std::optional<Tensor> rotary_cos,
      std::optional<Tensor> rotary_sin, std::optional<Tensor> cache_batch_idx,
      std::optional<Tensor> leftpad_k, std::optional<Tensor> block_table,
      std::optional<Tensor> alibi_slopes, Tensor out, float softmax_scale,
      bool is_causal, int64_t window_size_left, int64_t window_size_right,
      float softcap, bool is_rotary_interleaved,
      int64_t num_splits) const override {
    (void)softmax_scale;
    (void)is_causal;
    (void)window_size_left;
    (void)window_size_right;
    (void)is_rotary_interleaved;

    assert(!k.has_value() && !v.has_value() &&
           "`MhaFwdKvcache`: appending new `k` / `v` into cache is not "
           "supported by this Ascend path");
    assert(seqlens_k.has_value() && "`MhaFwdKvcache`: `seqlens_k` is required");
    assert(!rotary_cos.has_value() && !rotary_sin.has_value() &&
           "`MhaFwdKvcache`: rotary-on-append is not supported by this "
           "Ascend path");
    assert(!cache_batch_idx.has_value() &&
           "`MhaFwdKvcache`: `cache_batch_idx` is not supported by paged KV");
    assert(!leftpad_k.has_value() &&
           "`MhaFwdKvcache`: `leftpad_k` is not supported by this Ascend path");
    assert(block_table.has_value() &&
           "`MhaFwdKvcache`: paged KV `block_table` is required");
    assert(!alibi_slopes.has_value() &&
           "`MhaFwdKvcache`: `alibi_slopes` is not supported by this Ascend "
           "path");
    assert(softcap <= 0.0f &&
           "`MhaFwdKvcache`: `softcap` is not supported by this Ascend path");
    assert(num_splits == 0 &&
           "`MhaFwdKvcache`: split-KV is not supported by this Ascend path");

    auto stream = static_cast<aclrtStream>(stream_);
    auto seq_k = ascend::fia::CreateSeqLengths(seqlens_k.value(), stream);

    auto t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto t_out = out_cache_.get(out.data());
    auto t_block_table =
        block_table_cache_.get(const_cast<void*>(block_table->data()));

    auto t_k = aclCreateTensor(
        kv_shape_.data(), static_cast<int64_t>(kv_shape_.size()), kv_acl_dtype_,
        kv_strides_.data(), 0, ACL_FORMAT_ND, kv_storage_shape_.data(),
        static_cast<int64_t>(kv_storage_shape_.size()),
        const_cast<void*>(kcache.data()));
    auto t_v = aclCreateTensor(
        kv_shape_.data(), static_cast<int64_t>(kv_shape_.size()), kv_acl_dtype_,
        kv_strides_.data(), 0, ACL_FORMAT_ND, kv_storage_shape_.data(),
        static_cast<int64_t>(kv_storage_shape_.size()),
        const_cast<void*>(vcache.data()));

    const aclTensor* key_arr[] = {t_k};
    const aclTensor* value_arr[] = {t_v};
    auto key_list = aclCreateTensorList(key_arr, 1);
    auto value_list = aclCreateTensorList(value_arr, 1);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_q, key_list, value_list,
        nullptr,  // pseShift.
        nullptr,  // attenMask.
        nullptr, seq_k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, t_block_table, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, static_cast<int64_t>(num_heads_), softmax_scale_, 2147483647,
        2147483647, const_cast<char*>("BNSD"),
        static_cast<int64_t>(num_kv_heads_),
        0,  // sparseMode.
        0,  // innerPrecise.
        static_cast<int64_t>(page_block_size_),
        0,      // antiquantMode.
        false,  // softmaxLseFlag.
        0, 0, 0, t_out, nullptr, &ws_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "`aclnnFusedInferAttentionScoreV4GetWorkspaceSize` failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
    ret = aclnnFusedInferAttentionScoreV4(arena.buf, ws_size, executor, stream);
    assert(ret == ACL_SUCCESS && "`aclnnFusedInferAttentionScoreV4` failed");

    // Keep per-call descriptors alive until the RI graph task update using
    // them has completed.
    ascend::DeferOrRunAclCleanup([key_list, value_list, seq_k]() {
      aclDestroyTensorList(key_list);
      aclDestroyTensorList(value_list);
      aclDestroyIntArray(seq_k);
    });
  }

 private:
  mutable ascend::AclTensorCache q_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache block_table_cache_;

  std::vector<int64_t> kv_shape_;

  std::vector<int64_t> kv_strides_;

  std::vector<int64_t> kv_storage_shape_;

  aclDataType kv_acl_dtype_{ACL_DT_UNDEFINED};
};

}  // namespace infini::ops

#endif
