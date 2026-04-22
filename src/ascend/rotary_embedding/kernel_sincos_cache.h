#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_SINCOS_CACHE_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_SINCOS_CACHE_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_rope_with_sin_cos_cache.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// Rotary position embedding via `aclnnRopeWithSinCosCache` (implementation
// index 2).  This is the only Ascend fused rotary API that supports partial
// rotary (`rotary_dim < head_size`); it also natively supports both
// GPT-NeoX (`is_neox_style=true`) and GPT-J (`is_neox_style=false`) styles
// from the same interface.
//
// Input format: 2D contiguous `[num_tokens, num_heads * head_size]`.  The
// `aclnn` wrapper reads strides from the tensor descriptor — we pass a 2D
// descriptor even when the caller holds a 3D view `[T, N, D]`, since the
// memory layout is identical for contiguous tensors.  The 2D descriptor is
// what the `aclnn` sample in the CANN 8.5 docs uses.
//
// `cos_sin_cache` layout: `[max_seq_len, rotary_dim]` where the first
// `rotary_dim / 2` columns are cos and the next `rotary_dim / 2` are sin.
// The `aclnn` API splits internally via `cosSin.chunk(2, dim=-1)`.
//
// cf. `aclnn_rope_with_sin_cos_cache_hidden_attrs` memory: the public
// header hides four `REG_OP` attrs (`numQHeads`, `numKHeads`, `qStride`,
// `kStride`).  For 2D contiguous inputs the `aclnn` wrapper infers them
// correctly from the tensor descriptor; for 3D descriptors a previous
// attempt produced garbage output.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend, 2>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query,
           std::optional<Tensor> key, const Tensor cos_sin_cache,
           int64_t head_size, int64_t rotary_dim, bool is_neox_style,
           std::optional<Tensor> query_out = std::nullopt,
           std::optional<Tensor> key_out = std::nullopt,
           bool pre_gathered = false)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        rotary_dim, is_neox_style, query_out, key_out,
                        pre_gathered),
        max_seq_len_{cos_sin_cache.size(0)} {
    assert(has_key_ &&
           "Ascend `RotaryEmbedding` (`aclnnRopeWithSinCosCache`): `key` is "
           "required — this fused API always rotates Q and K together.");
    assert(!pre_gathered_ &&
           "Ascend `RotaryEmbedding` (`aclnnRopeWithSinCosCache`): "
           "`pre_gathered` is not supported — use implementation index 0 or "
           "1 for the pre-gathered fast path.");

    // Resolve optional out buffers (inplace on `query` / `key` when omitted).
    // Non-const so `.data()` returns a writable `void*`.
    Tensor q_out = query_out.value_or(query);
    Tensor k_out = key_out.value_or(*key);

    const int64_t num_tokens = num_tokens_;
    const int64_t num_q_heads = num_heads_;
    const int64_t num_kv_heads = num_kv_heads_;
    const int64_t head_dim = head_size_;
    aclDataType acl_dt = ascend::ToAclDtype(query.dtype());

    positions_cache_ = ascend::AclTensorCache(
        {num_tokens}, ACL_INT64, const_cast<void*>(positions.data()));
    q_in_cache_ =
        ascend::AclTensorCache({num_tokens, num_q_heads * head_dim}, acl_dt,
                               const_cast<void*>(query.data()));
    k_in_cache_ =
        ascend::AclTensorCache({num_tokens, num_kv_heads * head_dim}, acl_dt,
                               const_cast<void*>(key->data()));
    cos_sin_cache_cache_ =
        ascend::AclTensorCache({max_seq_len_, rotary_dim_}, acl_dt,
                               const_cast<void*>(cos_sin_cache.data()));
    q_out_cache_ = ascend::AclTensorCache({num_tokens, num_q_heads * head_dim},
                                          acl_dt, q_out.data());
    k_out_cache_ = ascend::AclTensorCache({num_tokens, num_kv_heads * head_dim},
                                          acl_dt, k_out.data());
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    positions_cache_.release();
    q_in_cache_.release();
    k_in_cache_.release();
    cos_sin_cache_cache_.release();
    q_out_cache_.release();
    k_out_cache_.release();
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor positions, const Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t head_size, int64_t rotary_dim, bool is_neox_style,
                  std::optional<Tensor> query_out,
                  std::optional<Tensor> key_out,
                  bool pre_gathered) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Resolve optional out buffers (inplace on `query` / `key` when omitted).
    Tensor q_out = query_out.value_or(query);
    Tensor k_out = key_out.value_or(*key);

    // Refresh cached descriptors with the current-call data pointers —
    // `Operator::call()` cache matches on shape/stride/dtype, so one
    // instance may serve multiple calls with different underlying buffers.
    auto t_pos = positions_cache_.get(const_cast<void*>(positions.data()));
    auto t_q = q_in_cache_.get(const_cast<void*>(query.data()));
    auto t_k = k_in_cache_.get(const_cast<void*>(key->data()));
    auto t_cache =
        cos_sin_cache_cache_.get(const_cast<void*>(cos_sin_cache.data()));
    auto t_q_out = q_out_cache_.get(const_cast<void*>(q_out.data()));
    auto t_k_out = k_out_cache_.get(const_cast<void*>(k_out.data()));

    // FIXME: per-call unbounded executor leak.  `aclnnRopeWithSinCosCache`'s
    // public header hides four `REG_OP` attrs (see
    // `aclnn_rope_with_sin_cos_cache_hidden_attrs` memory), so the official
    // `aclSetInputTensorAddr` index numbering for this kernel is not
    // documented — we cannot safely reuse a Repeatable executor across calls.
    // The async stream consumes the executor after enqueue, so destroying it
    // synchronously here races with the launch (SIGABRT).  Long-running
    // persistent workers (e.g. vLLM decode) accumulate one executor per
    // forward step until the runtime tears down.
    //
    // Resolve by obtaining the input-address index layout from the CANN team
    // (or deriving it from the binary) and switching to the cached-executor
    // pattern used in `kernel.h` / `kernel_atb.h`.
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnRopeWithSinCosCacheGetWorkspaceSize(
        t_pos, t_q, t_k, t_cache, /*mropeSection=*/nullptr, head_size,
        is_neox_style, t_q_out, t_k_out, &ws_size, &executor);
    assert(ret == 0 && "`aclnnRopeWithSinCosCacheGetWorkspaceSize` failed.");

    void* ws_buf = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_buf = arena.buf;
    }

    ret = aclnnRopeWithSinCosCache(ws_buf, ws_size, executor, stream);
    assert(ret == 0 && "`aclnnRopeWithSinCosCache` failed.");
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
