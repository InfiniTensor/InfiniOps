#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "ascend/atb_common_.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "base/reshape_and_cache.h"
#include "operator.h"

namespace infini::ops {

// ATB-based KV cache scatter via `atb::infer::ReshapeAndCacheParam`
// (implementation index 2).
//
// Handles both K and V in a single fused operation.  Profiled at ~9.5 us/call
// on Ascend 910B (256 tokens, fp16) — 3.7x faster than the
// `aclnnInplaceIndexCopy` path (index 0, ~35 us).
//
// The ATB operation is created once in the constructor.  Setup is called
// before each Execute to bind the VariantPack.
//
// NOTE: `ReshapeAndCacheParam` requires int32 `slot_mapping`.  When the
// caller passes int64 (the PyTorch / vLLM default), this operator issues an
// async `aclnnCast` to a pre-allocated int32 device buffer.  The cast
// executor is cached across calls and the whole step stays on the stream
// with no D2H/H2D round-trip, so the int64 path is NPUGraph-capturable and
// roughly on par with the int32 fast path.
//
// Input layout:
//   key, value : [num_tokens, num_kv_heads, head_size]
//   slot_mapping: [num_tokens] (int32 or int64)
//
// KV cache layout:
//   kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
//   Output key_cache = kv_cache[0], value_cache = kv_cache[1], each with
//   shape [num_blocks, block_size, num_kv_heads, head_size].
template <>
class Operator<ReshapeAndCache, Device::Type::kAscend, 2>
    : public ReshapeAndCache {
 public:
  Operator(const Tensor key, const Tensor value, const Tensor kv_cache,
           const Tensor slot_mapping, Tensor kv_cache_out)
      : ReshapeAndCache(key, value, kv_cache, slot_mapping, kv_cache_out) {
    auto num_blocks = static_cast<int64_t>(kv_cache.size(1));
    auto bs = static_cast<int64_t>(block_size_);
    int64_t nkv = static_cast<int64_t>(num_kv_heads_);
    int64_t hs = static_cast<int64_t>(head_size_);
    int64_t T = static_cast<int64_t>(num_tokens_);

    // Cache shapes for rebuilding VariantPack on each call.
    kv_shape_ = {num_blocks, bs, nkv, hs};
    key_shape_ = {T, nkv, hs};
    slot_shape_ = {T};
    acl_dt_ = ascend::toAclDtype(key.dtype());

    // Compute V-cache byte offset (kv_cache_out[1]).
    v_offset_bytes_ = static_cast<size_t>(kv_cache_out.stride(0)) *
                      kv_cache_out.element_size();

    // Element sizes for dataSize computation.
    elem_size_ = key.element_size();

    // Pre-allocate int32 device buffer for `slot_mapping`.
    // `ReshapeAndCacheParam` requires int32; int64 is silently ignored
    // (writes nothing).
    slot32_bytes_ = static_cast<size_t>(T) * sizeof(int32_t);
    aclrtMalloc(&slot32_buf_, slot32_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(slot32_buf_ && "aclrtMalloc for slot32_buf_ failed");

    slot_is_int32_ = (slot_mapping.element_size() == sizeof(int32_t));

    // Prepare aclnnCast descriptors for the int64 → int32 path.  Source
    // descriptor's data pointer is refreshed per call; destination is the
    // pre-allocated `slot32_buf_`.
    if (!slot_is_int32_) {
      slot_i64_cache_ = ascend::AclTensorCache(
          {T}, ACL_INT64, const_cast<void*>(slot_mapping.data()));
      slot_i32_cache_ = ascend::AclTensorCache({T}, ACL_INT32, slot32_buf_);
    }

    // Create the ATB operation (reused across calls).
    atb::infer::ReshapeAndCacheParam param;
    atb::Status s = atb::CreateOperation(param, &op_);
    assert(s == atb::NO_ERROR &&
           "atb::CreateOperation(ReshapeAndCache) failed");
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;
    if (op_) atb::DestroyOperation(op_);
    slot_i64_cache_.release();
    slot_i32_cache_.release();
    if (slot32_buf_) aclrtFree(slot32_buf_);
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor key, const Tensor value, const Tensor kv_cache,
                  const Tensor slot_mapping,
                  Tensor kv_cache_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // `ReshapeAndCacheParam` requires int32 `slot_mapping`.  When the
    // caller provides int64 (the PyTorch/vLLM default), issue an async
    // `aclnnCast` to the pre-allocated int32 device buffer — keeps the
    // whole step on-stream and NPUGraph-capturable.
    void* slot32_ptr;

    if (slot_is_int32_) {
      // Already int32 — pass through directly.
      slot32_ptr = const_cast<void*>(slot_mapping.data());
    } else {
      auto t_src = slot_i64_cache_.get(const_cast<void*>(slot_mapping.data()));
      auto t_dst = slot_i32_cache_.get(slot32_buf_);

      if (!cast_exec_) {
        aclnnCastGetWorkspaceSize(t_src, ACL_INT32, t_dst, &cast_ws_,
                                  &cast_exec_);
        aclSetAclOpExecutorRepeatable(cast_exec_);
      } else {
        aclSetInputTensorAddr(cast_exec_, 0, t_src,
                              const_cast<void*>(slot_mapping.data()));
        aclSetOutputTensorAddr(cast_exec_, 0, t_dst, slot32_buf_);
      }

      auto& cast_arena = ascend::GetWorkspacePool().Ensure(stream, cast_ws_);
      aclnnCast(cast_arena.buf, cast_ws_, cast_exec_, stream);
      slot32_ptr = slot32_buf_;
    }

    atb::Context* ctx = ascend::getAtbContext(stream);

    atb::VariantPack vp = buildVariantPack(const_cast<void*>(key.data()),
                                           const_cast<void*>(value.data()),
                                           kv_cache_out.data(), slot32_ptr);

    // Setup binds the VariantPack and computes workspace requirements.
    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);
    assert(s == atb::NO_ERROR &&
           "atb::Operation::Setup(ReshapeAndCache) failed");

    // Allocate workspace via the shared pool.
    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);
    assert(s == atb::NO_ERROR &&
           "atb::Operation::Execute(ReshapeAndCache) failed");
  }

 private:
  // Build the ATB VariantPack for this operation.
  //
  // ATB `ReshapeAndCache` expects 5 inputs and 2 outputs:
  //   inTensors[0] = key         [num_tokens, num_kv_heads, head_size]
  //   inTensors[1] = value       [num_tokens, num_kv_heads, head_size]
  //   inTensors[2] = key_cache   [num_blocks, block_size, num_kv_heads,
  //   head_size] inTensors[3] = value_cache [num_blocks, block_size,
  //   num_kv_heads, head_size] inTensors[4] = slot_mapping [num_tokens] (int32)
  //   outTensors[0] = key_cache   (same buffer, in-place)
  //   outTensors[1] = value_cache (same buffer, in-place)
  atb::VariantPack buildVariantPack(void* key_data, void* value_data,
                                    void* kv_out_data,
                                    void* slot32_data) const {
    int64_t num_tokens = key_shape_[0];
    int64_t nkv = key_shape_[1];
    int64_t hs = key_shape_[2];
    uint64_t kv_bytes =
        static_cast<uint64_t>(num_tokens * nkv * hs) * elem_size_;

    int64_t nb = kv_shape_[0];
    int64_t bs = kv_shape_[1];
    uint64_t cache_bytes =
        static_cast<uint64_t>(nb * bs * nkv * hs) * elem_size_;

    void* v_out_data = static_cast<char*>(kv_out_data) + v_offset_bytes_;

    atb::Tensor t_key =
        ascend::toAtbTensor(key_shape_, acl_dt_, key_data, kv_bytes);

    atb::Tensor t_value =
        ascend::toAtbTensor(key_shape_, acl_dt_, value_data, kv_bytes);

    atb::Tensor t_kv_k =
        ascend::toAtbTensor(kv_shape_, acl_dt_, kv_out_data, cache_bytes);

    atb::Tensor t_kv_v =
        ascend::toAtbTensor(kv_shape_, acl_dt_, v_out_data, cache_bytes);

    // Always int32 — the caller's `operator()` has already cast to int32.
    atb::Tensor t_slot =
        ascend::toAtbTensor(slot_shape_, ACL_INT32, slot32_data, slot32_bytes_);

    atb::VariantPack vp;
    vp.inTensors = {t_key, t_value, t_kv_k, t_kv_v, t_slot};
    vp.outTensors = {t_kv_k, t_kv_v};

    return vp;
  }

  atb::Operation* op_ = nullptr;

  std::vector<int64_t> kv_shape_;

  std::vector<int64_t> key_shape_;

  std::vector<int64_t> slot_shape_;

  aclDataType acl_dt_ = ACL_DT_UNDEFINED;

  size_t v_offset_bytes_ = 0;

  uint64_t elem_size_ = 0;

  // Pre-allocated int32 device buffer for `slot_mapping`.
  void* slot32_buf_ = nullptr;

  size_t slot32_bytes_ = 0;

  // True if the caller already provides int32 `slot_mapping`.
  bool slot_is_int32_ = false;

  // Cached aclnnCast descriptors (int64 slot_mapping → int32 buffer).
  mutable ascend::AclTensorCache slot_i64_cache_;

  mutable ascend::AclTensorCache slot_i32_cache_;

  mutable aclOpExecutor* cast_exec_ = nullptr;

  mutable uint64_t cast_ws_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_ATB_H_
