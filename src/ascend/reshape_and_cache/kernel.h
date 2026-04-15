#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_H_

#include <cassert>
#include <cstddef>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_index_copy.h"
#include "ascend/common.h"
#include "ascend/reshape_and_cache/registry.h"
#include "ascend/workspace_pool_.h"
#include "base/reshape_and_cache.h"
#include "operator.h"

namespace infini::ops {

// Device-side scatter via aclnnInplaceIndexCopy.
//
// The previous implementation copied slot_mapping D2H (aclrtSynchronizeStream),
// then issued per-token D2D memcpy in a host loop.  For batch=256, this meant
// ~100 us sync + ~500 us host loop overhead.  aclnnInplaceIndexCopy performs
// the scatter entirely on the NPU with two ACLNN calls (one for K, one for V),
// eliminating all D2H synchronisation and host-side loops.
//
// Requirement: slot_mapping must contain only non-negative values.  Padding
// tokens (slot < 0) must be filtered by the caller before invoking this
// operator.
template <>
class Operator<ReshapeAndCache, Device::Type::kAscend>
    : public ReshapeAndCache {
 public:
  Operator(const Tensor key, const Tensor value, const Tensor kv_cache,
           const Tensor slot_mapping, Tensor kv_cache_out)
      : ReshapeAndCache(key, value, kv_cache, slot_mapping, kv_cache_out),
        key_cache_(key),
        value_cache_(value),
        slot_cache_(slot_mapping) {
    auto num_blocks = static_cast<int64_t>(kv_cache.size(1));
    auto bs = static_cast<int64_t>(block_size_);
    int64_t total_slots = num_blocks * bs;
    int64_t nkv = static_cast<int64_t>(num_kv_heads_);
    int64_t hs = static_cast<int64_t>(head_size_);

    aclDataType acl_dt = ascend::toAclDtype(key.dtype());

    // Flattened K cache view: [total_slots, num_kv_heads, head_size].
    // K cache is kv_cache_out[0], starting at offset 0.
    kv_k_cache_ = ascend::AclTensorCache({total_slots, nkv, hs}, acl_dt,
                                         kv_cache_out.data());

    // V cache is kv_cache_out[1], offset by stride(0) elements.
    v_offset_bytes_ = static_cast<size_t>(kv_cache_out.stride(0)) *
                      kv_cache_out.element_size();
    kv_v_cache_ = ascend::AclTensorCache(
        {total_slots, nkv, hs}, acl_dt,
        static_cast<char*>(kv_cache_out.data()) + v_offset_bytes_);
  }

  void operator()(const Tensor key, const Tensor value, const Tensor kv_cache,
                  const Tensor slot_mapping,
                  Tensor kv_cache_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    void* kv_k_data = kv_cache_out.data();
    void* kv_v_data = static_cast<char*>(kv_cache_out.data()) + v_offset_bytes_;

    auto t_kv_k = kv_k_cache_.get(kv_k_data);
    auto t_kv_v = kv_v_cache_.get(kv_v_data);
    auto t_key = key_cache_.get(const_cast<void*>(key.data()));
    auto t_value = value_cache_.get(const_cast<void*>(value.data()));
    auto t_slot = slot_cache_.get(const_cast<void*>(slot_mapping.data()));

    // K cache scatter: kv_k[slot_mapping[i]] = key[i] along dim 0.
    // Executor caching is not used here because aclnnInplaceIndexCopy is an
    // inplace operation where self is both input and output; the executor
    // reuse via aclSetInputTensorAddr does not update the output reference.
    uint64_t k_ws = 0;
    aclOpExecutor* k_exec = nullptr;
    aclnnInplaceIndexCopyGetWorkspaceSize(t_kv_k, 0, t_slot, t_key, &k_ws,
                                          &k_exec);
    auto& k_arena = ascend::workspacePool().ensure(stream, k_ws);
    aclnnInplaceIndexCopy(k_arena.buf, k_ws, k_exec, stream);

    // V cache scatter: kv_v[slot_mapping[i]] = value[i] along dim 0.
    uint64_t v_ws = 0;
    aclOpExecutor* v_exec = nullptr;
    aclnnInplaceIndexCopyGetWorkspaceSize(t_kv_v, 0, t_slot, t_value, &v_ws,
                                          &v_exec);
    auto& v_arena = ascend::workspacePool().ensure(stream, v_ws);
    aclnnInplaceIndexCopy(v_arena.buf, v_ws, v_exec, stream);
  }

 private:
  mutable ascend::AclTensorCache kv_k_cache_;

  mutable ascend::AclTensorCache kv_v_cache_;

  mutable ascend::AclTensorCache key_cache_;

  mutable ascend::AclTensorCache value_cache_;

  mutable ascend::AclTensorCache slot_cache_;

  size_t v_offset_bytes_ = 0;
};

}  // namespace infini::ops

#endif
