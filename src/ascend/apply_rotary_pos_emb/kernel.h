#ifndef INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_KERNEL_H_
#define INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_KERNEL_H_

#include <cstddef>
#include <cstdint>

// clang-format off
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_v2.h"
// clang-format on
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/apply_rotary_pos_emb.h"
#include "operator.h"

namespace infini::ops {

// Apply-only rotary embedding via `aclnnApplyRotaryPosEmbV2` (CANN).
//
// Takes pre-gathered `[T, D]` cos/sin tensors directly — no `IndexSelect`.
// The caller is responsible for gathering from the full cos_sin_cache
// and expanding to neox format before calling this operator.
//
// V2 layout=4 (TND): Q `[T, Nq, D]`, K `[T, Nkv, D]`, cos/sin `[T, 1, D]`.
// Operates inplace on `query_out` and `key_out`.
//
// Restrictions:
//   - `is_neox_style` must be true (rotaryMode="half" only).
//   - fp16 only (V2 accumulation error is acceptable for inference).
template <>
class Operator<ApplyRotaryPosEmb, Device::Type::kAscend>
    : public ApplyRotaryPosEmb {
 public:
  Operator(const Tensor query, const Tensor key, const Tensor cos,
           const Tensor sin, int64_t head_size, bool is_neox_style,
           Tensor query_out, Tensor key_out)
      : ApplyRotaryPosEmb(query, key, cos, sin, head_size, is_neox_style,
                          query_out, key_out) {
    assert(is_neox_style &&
           "Ascend `ApplyRotaryPosEmb` requires neox style — "
           "aclnnApplyRotaryPosEmbV2 only supports rotaryMode \"half\"");

    const int64_t T = num_tokens_;
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    const int64_t D = head_size_;
    aclDataType acl_dt = ascend::toAclDtype(query.dtype());

    // V2 expects cos/sin as `[T, 1, D]`.  Input is `[T, D]` — same data,
    // different descriptor shape (T*1*D == T*D for contiguous tensors).
    cos_cache_ = ascend::AclTensorCache({T, 1, D}, acl_dt,
                                        const_cast<void*>(cos.data()));
    sin_cache_ = ascend::AclTensorCache({T, 1, D}, acl_dt,
                                        const_cast<void*>(sin.data()));
    q_cache_ = ascend::AclTensorCache({T, Nq, D}, acl_dt,
                                      const_cast<void*>(query_out.data()));
    k_cache_ = ascend::AclTensorCache({T, Nkv, D}, acl_dt,
                                      const_cast<void*>(key_out.data()));
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Release tensor caches — executors destroy their tensors internally.
    cos_cache_.release();
    sin_cache_.release();
    q_cache_.release();
    k_cache_.release();
  }

  void operator()(const Tensor query, const Tensor key, const Tensor cos,
                  const Tensor sin, int64_t head_size, bool is_neox_style,
                  Tensor query_out, Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    const int64_t T = query.size(0);
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    const int64_t D = head_size;

    // Copy q→q_out, k→k_out if not inplace (V2 operates inplace).
    size_t elem_sz = query.element_size();

    if (query.data() != query_out.data()) {
      aclrtMemcpyAsync(query_out.data(),
                       static_cast<size_t>(T * Nq * D) * elem_sz, query.data(),
                       static_cast<size_t>(T * Nq * D) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    if (key.data() != key_out.data()) {
      aclrtMemcpyAsync(key_out.data(),
                       static_cast<size_t>(T * Nkv * D) * elem_sz, key.data(),
                       static_cast<size_t>(T * Nkv * D) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    // Apply V2 RoPE inplace on q_out and k_out.
    auto t_cos = cos_cache_.get(const_cast<void*>(cos.data()));
    auto t_sin = sin_cache_.get(const_cast<void*>(sin.data()));
    auto t_q = q_cache_.get(query_out.data());
    auto t_k = k_cache_.get(key_out.data());

    if (!v2_exec_) {
      auto ws_ret = aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
          t_q, t_k, t_cos, t_sin, /*layout=*/4, const_cast<char*>("half"),
          &v2_ws_, &v2_exec_);
      assert(ws_ret == 0 && "aclnnApplyRotaryPosEmbV2GetWorkspaceSize failed");
      aclSetAclOpExecutorRepeatable(v2_exec_);
    } else {
      aclSetInputTensorAddr(v2_exec_, 0, t_q, query_out.data());
      aclSetInputTensorAddr(v2_exec_, 1, t_k, key_out.data());
      aclSetInputTensorAddr(v2_exec_, 2, t_cos, const_cast<void*>(cos.data()));
      aclSetInputTensorAddr(v2_exec_, 3, t_sin, const_cast<void*>(sin.data()));
    }

    auto& arena = ascend::workspacePool().ensure(stream, v2_ws_);
    auto exec_ret =
        aclnnApplyRotaryPosEmbV2(arena.buf, v2_ws_, v2_exec_, stream);
    assert(exec_ret == 0 && "aclnnApplyRotaryPosEmbV2 failed");
  }

 private:
  mutable ascend::AclTensorCache cos_cache_;

  mutable ascend::AclTensorCache sin_cache_;

  mutable ascend::AclTensorCache q_cache_;

  mutable ascend::AclTensorCache k_cache_;

  mutable aclOpExecutor* v2_exec_ = nullptr;

  mutable uint64_t v2_ws_ = 0;
};

}  // namespace infini::ops

#endif
