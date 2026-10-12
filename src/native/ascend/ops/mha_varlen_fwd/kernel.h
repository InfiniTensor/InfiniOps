#ifndef INFINI_OPS_ASCEND_MHA_VARLEN_FWD_KERNEL_H_
#define INFINI_OPS_ASCEND_MHA_VARLEN_FWD_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "base/mha_varlen_fwd.h"
#include "native/ascend/common.h"
#include "native/ascend/ops/fia_common_.h"
#include "native/ascend/ops/graph_cleanup_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<MhaVarlenFwd, Device::Type::kAscend> : public MhaVarlenFwd {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, Tensor out,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           std::optional<Tensor> seqused_k, std::optional<Tensor> leftpad_k,
           std::optional<Tensor> block_table,
           std::optional<Tensor> alibi_slopes, int64_t max_seqlen_q,
           int64_t max_seqlen_k, float p_dropout, float softmax_scale,
           bool zero_tensors, bool is_causal, int64_t window_size_left,
           int64_t window_size_right, float softcap, bool return_softmax,
           std::optional<int64_t> generator = std::nullopt,
           int64_t num_splits = 0)
      : MhaVarlenFwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k,
                     leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                     max_seqlen_k, p_dropout, softmax_scale, zero_tensors,
                     is_causal, window_size_left, window_size_right, softcap,
                     return_softmax, generator, num_splits),
        q_cache_(q),
        out_cache_(out) {
    ascend::fia::AssertSupportedDtype(q.dtype(), "`MhaVarlenFwd`");

    if (has_block_table_) {
      block_table_cache_ = ascend::AclTensorCache(block_table.value());

      const int64_t num_blocks = k.size(0);
      const int64_t block_size = k.size(1);
      const int64_t kv_nd = k.size(2) * k.size(3);
      kv_shape_ = {num_blocks, block_size, kv_nd};
      kv_strides_ = {block_size * kv_nd, kv_nd, 1};
      kv_storage_shape_ = {num_blocks * block_size * kv_nd};
      kv_acl_dtype_ = ascend::ToAclDtype(q.dtype());
      page_block_size_ = block_size;
      assert(page_block_size_ % 16 == 0 &&
             "`MhaVarlenFwd`: paged KV `block_size` must be 16-aligned for "
             "`float16` and `bfloat16`.");
    }

    if (is_causal_) {
      assert(max_seqlen_q_ <= 2048 && max_seqlen_k_ <= 2048 &&
             "`MhaVarlenFwd`: causal FIA mask currently supports "
             "`max_seqlen <= 2048`.");
      causal_mask_ = ascend::fia::MakeCausalMask(&causal_mask_buf_);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    if (causal_mask_) aclDestroyTensor(causal_mask_);
    if (causal_mask_buf_) aclrtFree(causal_mask_buf_);
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v, Tensor out,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  std::optional<Tensor> seqused_k,
                  std::optional<Tensor> leftpad_k,
                  std::optional<Tensor> block_table,
                  std::optional<Tensor> alibi_slopes, int64_t max_seqlen_q,
                  int64_t max_seqlen_k, float p_dropout, float softmax_scale,
                  bool zero_tensors, bool is_causal, int64_t window_size_left,
                  int64_t window_size_right, float softcap, bool return_softmax,
                  std::optional<int64_t> generator,
                  int64_t num_splits) const override {
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)softmax_scale;
    (void)is_causal;
    (void)window_size_left;
    (void)window_size_right;

    assert(!seqused_k.has_value() &&
           "`MhaVarlenFwd`: `seqused_k` is not supported by this Ascend path");
    assert(!leftpad_k.has_value() &&
           "`MhaVarlenFwd`: `leftpad_k` is not supported by this Ascend path");
    assert(block_table.has_value() == has_block_table_ &&
           "`MhaVarlenFwd`: `block_table` presence changed after descriptor "
           "creation");
    assert(!alibi_slopes.has_value() &&
           "`MhaVarlenFwd`: `alibi_slopes` is not supported by this Ascend "
           "path");
    assert(p_dropout == 0.0f &&
           "`MhaVarlenFwd`: dropout is not supported by this Ascend path");
    assert(!zero_tensors &&
           "`MhaVarlenFwd`: `zero_tensors` is not supported by this Ascend "
           "path");
    assert(softcap <= 0.0f &&
           "`MhaVarlenFwd`: `softcap` is not supported by this Ascend path");
    assert(!return_softmax &&
           "`MhaVarlenFwd`: returning softmax is not supported by InfiniOps "
           "in-place API");
    assert(!generator.has_value() &&
           "`MhaVarlenFwd`: `generator` is only meaningful for dropout");
    assert(num_splits == 0 &&
           "`MhaVarlenFwd`: split-KV is not supported by this Ascend path");

    auto stream = static_cast<aclrtStream>(stream_);
    int64_t sparse_mode = 0;
    int64_t pre_tokens = 2147483647;
    int64_t next_tokens = 2147483647;
    ascend::fia::ResolveSparseMode(is_causal_, window_size_left_,
                                   window_size_right_, sparse_mode, pre_tokens,
                                   next_tokens);

    auto seq_q = ascend::fia::CreateCumSeqLengths(cu_seqlens_q, stream);
    auto seq_k = has_block_table_
                     ? ascend::fia::CreateDiffSeqLengths(cu_seqlens_k, stream)
                     : ascend::fia::CreateCumSeqLengths(cu_seqlens_k, stream);

    auto t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto t_out = out_cache_.get(out.data());
    aclTensor* t_block_table = nullptr;
    aclTensor* t_k = nullptr;
    aclTensor* t_v = nullptr;

    if (has_block_table_) {
      assert(page_block_size_ == static_cast<int64_t>(k.size(1)) &&
             "`MhaVarlenFwd`: paged KV `block_size` changed after descriptor "
             "creation");
      t_block_table =
          block_table_cache_.get(const_cast<void*>(block_table->data()));
      t_k = aclCreateTensor(kv_shape_.data(),
                            static_cast<int64_t>(kv_shape_.size()),
                            kv_acl_dtype_, kv_strides_.data(), 0, ACL_FORMAT_ND,
                            kv_storage_shape_.data(),
                            static_cast<int64_t>(kv_storage_shape_.size()),
                            const_cast<void*>(k.data()));
      t_v = aclCreateTensor(kv_shape_.data(),
                            static_cast<int64_t>(kv_shape_.size()),
                            kv_acl_dtype_, kv_strides_.data(), 0, ACL_FORMAT_ND,
                            kv_storage_shape_.data(),
                            static_cast<int64_t>(kv_storage_shape_.size()),
                            const_cast<void*>(v.data()));
    } else {
      t_k = ascend::BuildAclTensor(k);
      t_v = ascend::BuildAclTensor(v);
    }

    const aclTensor* key_arr[] = {t_k};
    const aclTensor* value_arr[] = {t_v};
    auto key_list = aclCreateTensorList(key_arr, 1);
    auto value_list = aclCreateTensorList(value_arr, 1);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_q, key_list, value_list,
        nullptr,       // pseShift.
        causal_mask_,  // attenMask.
        seq_q, seq_k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, t_block_table, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, static_cast<int64_t>(num_heads_), softmax_scale_, pre_tokens,
        next_tokens, const_cast<char*>("TND"),
        static_cast<int64_t>(num_kv_heads_), sparse_mode,
        0,                 // innerPrecise.
        page_block_size_,  // blockSize.
        0,                 // antiquantMode.
        false,             // softmaxLseFlag.
        0, 0, 0, t_out, nullptr, &ws_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "`aclnnFusedInferAttentionScoreV4GetWorkspaceSize` failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
    ret = aclnnFusedInferAttentionScoreV4(arena.buf, ws_size, executor, stream);
    assert(ret == ACL_SUCCESS && "`aclnnFusedInferAttentionScoreV4` failed");

    ascend::DeferOrRunAclCleanup([key_list, value_list, seq_q, seq_k]() {
      aclDestroyTensorList(key_list);
      aclDestroyTensorList(value_list);
      aclDestroyIntArray(seq_q);
      aclDestroyIntArray(seq_k);
    });
  }

 private:
  mutable ascend::AclTensorCache q_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache block_table_cache_;

  aclTensor* causal_mask_ = nullptr;

  void* causal_mask_buf_ = nullptr;

  std::vector<int64_t> kv_shape_;

  std::vector<int64_t> kv_strides_;

  std::vector<int64_t> kv_storage_shape_;

  aclDataType kv_acl_dtype_{ACL_DT_UNDEFINED};

  int64_t page_block_size_{0};
};

}  // namespace infini::ops

#endif
