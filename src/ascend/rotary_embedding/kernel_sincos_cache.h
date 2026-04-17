#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_SINCOS_CACHE_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_SINCOS_CACHE_H_

#include <cassert>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_rope_with_sin_cos_cache.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// Rotary position embedding via `aclnnRopeWithSinCosCache` (implementation
// index 2). This is the only Ascend fused rotary API that supports partial
// rotary (`rotary_dim < head_size`); it also natively supports both
// GPT-NeoX (`is_neox_style=true`) and GPT-J (`is_neox_style=false`) styles
// from the same interface.
//
// Input format: 2D contiguous `[num_tokens, num_heads * head_size]`. The
// aclnn wrapper reads strides from the tensor descriptor — we pass a 2D
// descriptor even when the caller holds a 3D view `[T, N, D]`, since the
// memory layout is identical for contiguous tensors. The 2D descriptor is
// what the aclnn sample in the CANN 8.5 docs uses.
//
// `cos_sin_cache` layout: `[max_seq_len, rotary_dim]` where the first
// `rotary_dim / 2` columns are cos and the next `rotary_dim / 2` are sin.
// The aclnn API splits internally via `cosSin.chunk(2, dim=-1)`.
//
// cf. `aclnn_rope_with_sin_cos_cache_hidden_attrs` memory: the public
// header hides four `REG_OP` attrs (`numQHeads`, `numKHeads`, `qStride`,
// `kStride`). For 2D contiguous inputs the aclnn wrapper infers them
// correctly from the tensor descriptor; for 3D descriptors a previous
// attempt produced garbage output.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend, 2>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query, const Tensor key,
           const Tensor cos_sin_cache, int64_t head_size, int64_t rotary_dim,
           bool is_neox_style, Tensor query_out, Tensor key_out)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        rotary_dim, is_neox_style, query_out, key_out),
        max_seq_len_{cos_sin_cache.size(0)} {
    const int64_t T = num_tokens_;
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    const int64_t D = head_size_;
    aclDataType acl_dt = ascend::toAclDtype(query.dtype());

    positions_cache_ = ascend::AclTensorCache(
        {T}, ACL_INT64, const_cast<void*>(positions.data()));
    q_in_cache_ = ascend::AclTensorCache(
        {T, Nq * D}, acl_dt, const_cast<void*>(query.data()));
    k_in_cache_ = ascend::AclTensorCache(
        {T, Nkv * D}, acl_dt, const_cast<void*>(key.data()));
    cos_sin_cache_cache_ = ascend::AclTensorCache(
        {max_seq_len_, rotary_dim_}, acl_dt,
        const_cast<void*>(cos_sin_cache.data()));
    q_out_cache_ =
        ascend::AclTensorCache({T, Nq * D}, acl_dt, query_out.data());
    k_out_cache_ =
        ascend::AclTensorCache({T, Nkv * D}, acl_dt, key_out.data());
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    positions_cache_.release();
    q_in_cache_.release();
    k_in_cache_.release();
    cos_sin_cache_cache_.release();
    q_out_cache_.release();
    k_out_cache_.release();
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Refresh cached descriptors with the current-call data pointers —
    // `Operator::call()` cache matches on shape/stride/dtype, so one
    // instance may serve multiple calls with different underlying buffers.
    auto t_pos = positions_cache_.get(const_cast<void*>(positions.data()));
    auto t_q = q_in_cache_.get(const_cast<void*>(query.data()));
    auto t_k = k_in_cache_.get(const_cast<void*>(key.data()));
    auto t_cache =
        cos_sin_cache_cache_.get(const_cast<void*>(cos_sin_cache.data()));
    auto t_q_out = q_out_cache_.get(query_out.data());
    auto t_k_out = k_out_cache_.get(key_out.data());

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnRopeWithSinCosCacheGetWorkspaceSize(
        t_pos, t_q, t_k, t_cache, /*mropeSection=*/nullptr, head_size,
        is_neox_style, t_q_out, t_k_out, &ws_size, &executor);
    assert(ret == 0 && "aclnnRopeWithSinCosCacheGetWorkspaceSize failed");

    void* ws_buf = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_buf = arena.buf;
    }

    ret = aclnnRopeWithSinCosCache(ws_buf, ws_size, executor, stream);
    assert(ret == 0 && "aclnnRopeWithSinCosCache failed");
  }

 private:
  int64_t max_seq_len_;

  mutable ascend::AclTensorCache positions_cache_;

  mutable ascend::AclTensorCache q_in_cache_;

  mutable ascend::AclTensorCache k_in_cache_;

  mutable ascend::AclTensorCache cos_sin_cache_cache_;

  mutable ascend::AclTensorCache q_out_cache_;

  mutable ascend::AclTensorCache k_out_cache_;
};

}  // namespace infini::ops

#endif
