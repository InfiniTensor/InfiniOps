#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_v2.h"
#include "aclnnop/aclnn_index_select.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// Rotary position embedding via aclnnApplyRotaryPosEmbV2.
//
// V2 handles Q and K simultaneously in a single inplace call (layout=4, TND).
// The `rotaryMode` parameter accepts "half", "interleave", or "quarter", but
// CANN currently only supports "half" (neox style).  Passing "interleave" or
// "quarter" returns ACLNN_ERR_PARAM_INVALID.
//
// fp16 note: V2 accumulates with ~4 ULP error for float16 (max diff ~0.008),
// which exceeds strict atol=0.001 tests but is acceptable for inference.
// bfloat16 passes with atol=0.005.
//
// Restrictions:
//   - rotary_dim must equal head_size (partial rotation not supported).
//   - is_neox_style must be true (rotaryMode="half" only).
// All mainstream models (LLaMA, Qwen, Mistral, DeepSeek) satisfy both.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query, const Tensor key,
           const Tensor cos_sin_cache, int64_t head_size, int64_t rotary_dim,
           bool is_neox_style, Tensor query_out, Tensor key_out)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        rotary_dim, is_neox_style, query_out, key_out),
        max_seq_len_{cos_sin_cache.size(0)},
        elem_sz_{cos_sin_cache.element_size()} {
    assert(rotary_dim == head_size &&
           "Ascend `RotaryEmbedding` requires rotary_dim == head_size "
           "(partial rotation not supported)");
    assert(is_neox_style &&
           "Ascend `RotaryEmbedding` requires neox style — "
           "aclnnApplyRotaryPosEmbV2 rotaryMode only supports \"half\"; "
           "\"interleave\" and \"quarter\" return ACLNN_ERR_PARAM_INVALID");

    const int64_t D = head_size_;
    size_t table_bytes = static_cast<size_t>(max_seq_len_ * D) * elem_sz_;

    // Allocate device buffers for expanded cos/sin tables [max_seq_len, D].
    aclrtMalloc(&cos_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Upload initial cos_sin_cache.
    uploadCosSinCache(cos_sin_cache);

    const int64_t T = num_tokens_;
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    aclDataType acl_dt = ascend::toAclDtype(query.dtype());

    // Gathered cos/sin buffers [T, D] — filled by aclnnIndexSelect each call.
    size_t gathered_bytes = static_cast<size_t>(T * D) * elem_sz_;
    aclrtMalloc(&cos_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // IndexSelect descriptors: table ptrs stable, positions ptr varies.
    cos_table_cache_ = ascend::AclTensorCache(
        {max_seq_len_, D}, acl_dt, cos_table_dev_);
    sin_table_cache_ = ascend::AclTensorCache(
        {max_seq_len_, D}, acl_dt, sin_table_dev_);
    idx_cache_ = ascend::AclTensorCache(
        {T}, ACL_INT64, const_cast<void*>(positions.data()));
    cos_out_cache_ = ascend::AclTensorCache({T, D}, acl_dt, cos_dev_);
    sin_out_cache_ = ascend::AclTensorCache({T, D}, acl_dt, sin_dev_);

    // V2 descriptors: cos/sin [T, 1, D], Q [T, Nq, D], K [T, Nkv, D].
    cos_v2_cache_ = ascend::AclTensorCache({T, 1, D}, acl_dt, cos_dev_);
    sin_v2_cache_ = ascend::AclTensorCache({T, 1, D}, acl_dt, sin_dev_);
    q_cache_ = ascend::AclTensorCache(
        {T, Nq, D}, acl_dt, const_cast<void*>(query_out.data()));
    k_cache_ = ascend::AclTensorCache(
        {T, Nkv, D}, acl_dt, const_cast<void*>(key_out.data()));
  }

  ~Operator() {
    if (idx_cos_exec_) aclDestroyAclOpExecutor(idx_cos_exec_);
    if (idx_sin_exec_) aclDestroyAclOpExecutor(idx_sin_exec_);
    if (v2_exec_) aclDestroyAclOpExecutor(v2_exec_);

    if (cos_table_dev_) aclrtFree(cos_table_dev_);
    if (sin_table_dev_) aclrtFree(sin_table_dev_);
    if (cos_dev_) aclrtFree(cos_dev_);
    if (sin_dev_) aclrtFree(sin_dev_);
  }

  void operator()(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Re-upload if cos_sin_cache data pointer changed (different tensor).
    // In production this pointer is stable (fixed weight), so this is a no-op.
    if (cos_sin_cache.data() != last_cos_sin_ptr_) {
      uploadCosSinCache(cos_sin_cache);
    }

    const int64_t T = query.size(0);
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    const int64_t D = head_size;

    // Step 1: Gather cos/sin by positions via aclnnIndexSelect (async).
    {
      auto t_cos_table = cos_table_cache_.get(cos_table_dev_);
      auto t_sin_table = sin_table_cache_.get(sin_table_dev_);
      auto t_idx = idx_cache_.get(const_cast<void*>(positions.data()));
      auto t_cos_out = cos_out_cache_.get(cos_dev_);
      auto t_sin_out = sin_out_cache_.get(sin_dev_);

      if (!idx_cos_exec_) {
        aclnnIndexSelectGetWorkspaceSize(t_cos_table, 0, t_idx, t_cos_out,
                                         &idx_cos_ws_, &idx_cos_exec_);
        aclSetAclOpExecutorRepeatable(idx_cos_exec_);
      } else {
        aclSetInputTensorAddr(idx_cos_exec_, 1, t_idx,
                              const_cast<void*>(positions.data()));
      }

      if (!idx_sin_exec_) {
        aclnnIndexSelectGetWorkspaceSize(t_sin_table, 0, t_idx, t_sin_out,
                                         &idx_sin_ws_, &idx_sin_exec_);
        aclSetAclOpExecutorRepeatable(idx_sin_exec_);
      } else {
        aclSetInputTensorAddr(idx_sin_exec_, 1, t_idx,
                              const_cast<void*>(positions.data()));
      }

      uint64_t ws_max = idx_cos_ws_ > idx_sin_ws_ ? idx_cos_ws_ : idx_sin_ws_;
      auto& arena = ascend::workspacePool().ensure(stream, ws_max);

      aclnnIndexSelect(arena.buf, idx_cos_ws_, idx_cos_exec_, stream);
      aclnnIndexSelect(arena.buf, idx_sin_ws_, idx_sin_exec_, stream);
    }

    // Step 2: Copy q→q_out, k→k_out if not inplace (V2 operates inplace).
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

    // Step 3: Apply V2 RoPE inplace on q_out and k_out.
    auto t_cos = cos_v2_cache_.get(cos_dev_);
    auto t_sin = sin_v2_cache_.get(sin_dev_);
    auto t_q = q_cache_.get(query_out.data());
    auto t_k = k_cache_.get(key_out.data());

    if (!v2_exec_) {
      aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
          t_q, t_k, t_cos, t_sin, /*layout=*/4, const_cast<char*>("half"),
          &v2_ws_, &v2_exec_);
      aclSetAclOpExecutorRepeatable(v2_exec_);
    } else {
      aclSetInputTensorAddr(v2_exec_, 0, t_q, query_out.data());
      aclSetInputTensorAddr(v2_exec_, 1, t_k, key_out.data());
    }

    auto& arena = ascend::workspacePool().ensure(stream, v2_ws_);
    aclnnApplyRotaryPosEmbV2(arena.buf, v2_ws_, v2_exec_, stream);
  }

 private:
  // D2H copy cos_sin_cache, split into cos/sin, neox-expand, and upload to
  // device.  Called once at construction and again if the caller provides a
  // different cos_sin_cache tensor (detected by data-pointer change).
  void uploadCosSinCache(const Tensor cos_sin_cache) const {
    const int64_t D = head_size_;
    const int64_t half_D = D / 2;
    size_t table_bytes = static_cast<size_t>(max_seq_len_ * D) * elem_sz_;

    std::vector<uint8_t> cache_host(table_bytes);
    aclrtMemcpy(cache_host.data(), table_bytes, cos_sin_cache.data(),
                table_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<uint8_t> cos_host(table_bytes);
    std::vector<uint8_t> sin_host(table_bytes);

    for (int64_t p = 0; p < max_seq_len_; ++p) {
      for (int64_t j = 0; j < half_D; ++j) {
        const auto* c_src =
            cache_host.data() +
            static_cast<size_t>(p * D + j) * elem_sz_;
        const auto* s_src =
            cache_host.data() +
            static_cast<size_t>(p * D + half_D + j) * elem_sz_;

        std::memcpy(
            cos_host.data() + static_cast<size_t>(p * D + j) * elem_sz_,
            c_src, elem_sz_);
        std::memcpy(
            cos_host.data() +
                static_cast<size_t>(p * D + half_D + j) * elem_sz_,
            c_src, elem_sz_);
        std::memcpy(
            sin_host.data() + static_cast<size_t>(p * D + j) * elem_sz_,
            s_src, elem_sz_);
        std::memcpy(
            sin_host.data() +
                static_cast<size_t>(p * D + half_D + j) * elem_sz_,
            s_src, elem_sz_);
      }
    }

    aclrtMemcpy(cos_table_dev_, table_bytes, cos_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes, sin_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);

    last_cos_sin_ptr_ = cos_sin_cache.data();
  }

  int64_t max_seq_len_;

  size_t elem_sz_;

  // Tracks which cos_sin_cache was last uploaded so we can re-upload if the
  // caller provides a different tensor (same shape, different data).
  mutable const void* last_cos_sin_ptr_ = nullptr;

  // Pre-expanded cos/sin tables on device: [max_seq_len, D].
  void* cos_table_dev_ = nullptr;

  void* sin_table_dev_ = nullptr;

  // Device buffers for gathered [T, D] cos/sin.
  void* cos_dev_ = nullptr;

  void* sin_dev_ = nullptr;

  // IndexSelect descriptors.
  mutable ascend::AclTensorCache cos_table_cache_;

  mutable ascend::AclTensorCache sin_table_cache_;

  mutable ascend::AclTensorCache idx_cache_;

  mutable ascend::AclTensorCache cos_out_cache_;

  mutable ascend::AclTensorCache sin_out_cache_;

  // V2 descriptors.
  mutable ascend::AclTensorCache cos_v2_cache_;

  mutable ascend::AclTensorCache sin_v2_cache_;

  mutable ascend::AclTensorCache q_cache_;

  mutable ascend::AclTensorCache k_cache_;

  // Cached executors.
  mutable aclOpExecutor* idx_cos_exec_ = nullptr;

  mutable uint64_t idx_cos_ws_ = 0;

  mutable aclOpExecutor* idx_sin_exec_ = nullptr;

  mutable uint64_t idx_sin_ws_ = 0;

  mutable aclOpExecutor* v2_exec_ = nullptr;

  mutable uint64_t v2_ws_ = 0;
};

}  // namespace infini::ops

#endif
