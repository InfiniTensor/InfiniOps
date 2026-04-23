#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <optional>
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

// Rotary position embedding via `aclnnApplyRotaryPosEmbV2`.
//
// V2 handles Q and K simultaneously in a single inplace call (`layout=4`,
// TND).  When `pre_gathered` is true, `cos_sin_cache` is interpreted as the
// already-gathered `[T, head_size * 2]` neox-expanded table and the internal
// `aclnnIndexSelect` step is skipped.
//
// fp16 note: V2 accumulates with ~4 ULP error for float16 (max diff ~0.008),
// which exceeds strict atol=0.001 tests but is acceptable for inference.
// bfloat16 passes with atol=0.005.
//
// Restrictions (implementation choices, not V2 API limits):
//   - `rotary_dim` must equal `head_size` (partial rotation not
//     implemented; V2's cos/sin second dim can be `head_size / 2` per the
//     CANN 8.5 docs).
//   - `is_neox_style` must be `true`.  V2 accepts `rotaryMode="half" /
//     "interleave" / "quarter"`; this wrapper plumbs only `"half"`.
// All mainstream models (LLaMA, Qwen, Mistral, DeepSeek) satisfy both.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query,
           std::optional<Tensor> key, int64_t head_size,
           const Tensor cos_sin_cache, bool is_neox_style, int64_t rotary_dim,
           std::optional<Tensor> query_out = std::nullopt,
           std::optional<Tensor> key_out = std::nullopt,
           bool pre_gathered = false)
      : RotaryEmbedding(positions, query, key, head_size, cos_sin_cache,
                        is_neox_style, rotary_dim, query_out, key_out,
                        pre_gathered),
        max_seq_len_{cos_sin_cache.size(0)},
        elem_sz_{cos_sin_cache.element_size()} {
    assert(rotary_dim == head_size &&
           "Ascend `RotaryEmbedding`: `rotary_dim` must equal `head_size` "
           "(partial rotation is not implemented in this wrapper).");
    assert(is_neox_style &&
           "Ascend `RotaryEmbedding`: `is_neox_style` must be `true` — "
           "this wrapper only plumbs `rotaryMode=\"half\"` through "
           "`aclnnApplyRotaryPosEmbV2`.");
    assert(has_key_ &&
           "Ascend `RotaryEmbedding` (impl 0): `key` is required — "
           "`aclnnApplyRotaryPosEmbV2` always rotates Q and K together.");

    // Resolve optional out buffers; when omitted, RoPE writes back in place
    // on `query` / `key` — vLLM-style inplace semantics.
    Tensor q_out = query_out.value_or(query);
    Tensor k_out = key_out.value_or(*key);

    const int64_t head_dim = head_size_;
    const int64_t num_tokens = num_tokens_;
    const int64_t num_q_heads = num_heads_;
    const int64_t num_kv_heads = num_kv_heads_;
    aclDataType acl_dt = ascend::ToAclDtype(query.dtype());

    if (!pre_gathered_) {
      // Full cache path: allocate expanded cos/sin tables of
      // `[max_seq_len, head_dim]`, and `[T, head_dim]` gathered buffers that
      // `aclnnIndexSelect` writes per call.
      size_t table_bytes =
          static_cast<size_t>(max_seq_len_ * head_dim) * elem_sz_;

      aclrtMalloc(&cos_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      aclrtMalloc(&sin_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

      // Upload the initial cos_sin_cache.  `cos_sin_cache_data_` memorizes
      // the source pointer; if the caller later hands in a different buffer,
      // `operator()` re-runs the upload.
      UploadCosSinCache(cos_sin_cache);
      cos_sin_cache_data_ = cos_sin_cache.data();

      size_t gathered_bytes =
          static_cast<size_t>(num_tokens * head_dim) * elem_sz_;
      aclrtMalloc(&cos_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      aclrtMalloc(&sin_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

      // IndexSelect descriptors: table ptrs stable, positions ptr varies.
      cos_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                                acl_dt, cos_table_dev_);
      sin_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                                acl_dt, sin_table_dev_);
      idx_cache_ = ascend::AclTensorCache({num_tokens}, ACL_INT64,
                                          const_cast<void*>(positions.data()));
      cos_out_cache_ =
          ascend::AclTensorCache({num_tokens, head_dim}, acl_dt, cos_dev_);
      sin_out_cache_ =
          ascend::AclTensorCache({num_tokens, head_dim}, acl_dt, sin_dev_);
    }

    // V2 descriptors: cos/sin `[T, 1, head_dim]`, Q `[T, Nq, head_dim]`,
    // K `[T, Nkv, head_dim]`.  When `pre_gathered` is true, cos/sin point
    // into the caller's `cos_sin_cache`: row 0..T-1 is cos, row T..2T-1 is
    // sin (stacked along dim=0 by the shim).
    void* cos_init = cos_dev_;
    void* sin_init = sin_dev_;

    if (pre_gathered_) {
      auto* base =
          static_cast<uint8_t*>(const_cast<void*>(cos_sin_cache.data()));
      cos_init = base;
      sin_init = base + static_cast<size_t>(num_tokens * head_dim) * elem_sz_;
    }

    cos_v2_cache_ =
        ascend::AclTensorCache({num_tokens, 1, head_dim}, acl_dt, cos_init);
    sin_v2_cache_ =
        ascend::AclTensorCache({num_tokens, 1, head_dim}, acl_dt, sin_init);
    q_cache_ = ascend::AclTensorCache({num_tokens, num_q_heads, head_dim},
                                      acl_dt, const_cast<void*>(q_out.data()));
    k_cache_ = ascend::AclTensorCache({num_tokens, num_kv_heads, head_dim},
                                      acl_dt, const_cast<void*>(k_out.data()));
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    cos_table_cache_.release();
    sin_table_cache_.release();
    idx_cache_.release();
    cos_out_cache_.release();
    sin_out_cache_.release();
    cos_v2_cache_.release();
    sin_v2_cache_.release();
    q_cache_.release();
    k_cache_.release();

    if (cos_table_dev_) aclrtFree(cos_table_dev_);
    if (sin_table_dev_) aclrtFree(sin_table_dev_);
    if (cos_dev_) aclrtFree(cos_dev_);
    if (sin_dev_) aclrtFree(sin_dev_);
  }

  void operator()(const Tensor positions, const Tensor query,
                  std::optional<Tensor> key, int64_t head_size,
                  const Tensor cos_sin_cache, bool is_neox_style,
                  int64_t rotary_dim, std::optional<Tensor> query_out,
                  std::optional<Tensor> key_out,
                  bool pre_gathered) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Resolve optional out buffers (inplace on `query` / `key` when omitted).
    // Non-const so `.data()` returns a writable `void*`.
    Tensor q_out = query_out.value_or(query);
    Tensor k_out = key_out.value_or(*key);

    const int64_t num_tokens = query.size(0);
    const int64_t num_q_heads = num_heads_;
    const int64_t num_kv_heads = num_kv_heads_;
    const int64_t head_dim = head_size;

    const void* cos_sin_for_v2 = nullptr;
    const void* sin_for_v2 = nullptr;

    if (!pre_gathered) {
      // `CacheKey` matches on shape/stride/dtype and ignores data pointers,
      // so a cached operator instance may be reused across calls that hand in
      // different `cos_sin_cache` allocations.  Re-upload when the source
      // pointer changes.  See `operator_cache_stale_data` in memory.
      if (cos_sin_cache.data() != cos_sin_cache_data_) {
        UploadCosSinCache(cos_sin_cache);
        cos_sin_cache_data_ = cos_sin_cache.data();
      }

      // Step 1: Gather cos/sin by positions via `aclnnIndexSelect` (async).
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
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_max);

      aclnnIndexSelect(arena.buf, idx_cos_ws_, idx_cos_exec_, stream);
      aclnnIndexSelect(arena.buf, idx_sin_ws_, idx_sin_exec_, stream);

      cos_sin_for_v2 = cos_dev_;
      sin_for_v2 = sin_dev_;
    } else {
      // Pre-gathered: caller passes `[2 * T, head_size]` — rows 0..T-1 are
      // neox-expanded cos, rows T..2T-1 are neox-expanded sin (stacked via
      // `torch.cat([cos, sin], dim=0)`).
      const auto* base = static_cast<const uint8_t*>(cos_sin_cache.data());
      cos_sin_for_v2 = base;
      sin_for_v2 = base + static_cast<size_t>(num_tokens * head_dim) * elem_sz_;
    }

    // Step 2: Copy q -> q_out, k -> k_out if not inplace (V2 operates
    // inplace).
    size_t elem_sz = query.element_size();

    if (query.data() != q_out.data()) {
      aclrtMemcpyAsync(
          q_out.data(),
          static_cast<size_t>(num_tokens * num_q_heads * head_dim) * elem_sz,
          query.data(),
          static_cast<size_t>(num_tokens * num_q_heads * head_dim) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    if (key->data() != k_out.data()) {
      aclrtMemcpyAsync(
          k_out.data(),
          static_cast<size_t>(num_tokens * num_kv_heads * head_dim) * elem_sz,
          key->data(),
          static_cast<size_t>(num_tokens * num_kv_heads * head_dim) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    // Step 3: Apply V2 RoPE inplace on q_out and k_out.
    auto t_cos = cos_v2_cache_.get(const_cast<void*>(cos_sin_for_v2));
    auto t_sin = sin_v2_cache_.get(const_cast<void*>(sin_for_v2));
    auto t_q = q_cache_.get(q_out.data());
    auto t_k = k_cache_.get(k_out.data());

    if (!v2_exec_) {
      aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
          t_q, t_k, t_cos, t_sin, /*layout=*/4, const_cast<char*>("half"),
          &v2_ws_, &v2_exec_);
      aclSetAclOpExecutorRepeatable(v2_exec_);
    } else {
      aclSetInputTensorAddr(v2_exec_, 0, t_q, q_out.data());
      aclSetInputTensorAddr(v2_exec_, 1, t_k, k_out.data());
      aclSetInputTensorAddr(v2_exec_, 2, t_cos,
                            const_cast<void*>(cos_sin_for_v2));
      aclSetInputTensorAddr(v2_exec_, 3, t_sin, const_cast<void*>(sin_for_v2));
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, v2_ws_);
    aclnnApplyRotaryPosEmbV2(arena.buf, v2_ws_, v2_exec_, stream);
  }

 private:
  // D2H copy `cos_sin_cache`, split into cos/sin, neox-expand, and upload to
  // device.  Called at construction and on cache-pointer change.
  void UploadCosSinCache(const Tensor cos_sin_cache) const {
    const int64_t head_dim = head_size_;
    const int64_t half_head_dim = head_dim / 2;
    size_t table_bytes =
        static_cast<size_t>(max_seq_len_ * head_dim) * elem_sz_;

    std::vector<uint8_t> cache_host(table_bytes);
    aclrtMemcpy(cache_host.data(), table_bytes, cos_sin_cache.data(),
                table_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<uint8_t> cos_host(table_bytes);
    std::vector<uint8_t> sin_host(table_bytes);

    for (int64_t p = 0; p < max_seq_len_; ++p) {
      for (int64_t j = 0; j < half_head_dim; ++j) {
        const auto* c_src = cache_host.data() +
                            static_cast<size_t>(p * head_dim + j) * elem_sz_;
        const auto* s_src =
            cache_host.data() +
            static_cast<size_t>(p * head_dim + half_head_dim + j) * elem_sz_;

        std::memcpy(
            cos_host.data() + static_cast<size_t>(p * head_dim + j) * elem_sz_,
            c_src, elem_sz_);
        std::memcpy(cos_host.data() +
                        static_cast<size_t>(p * head_dim + half_head_dim + j) *
                            elem_sz_,
                    c_src, elem_sz_);
        std::memcpy(
            sin_host.data() + static_cast<size_t>(p * head_dim + j) * elem_sz_,
            s_src, elem_sz_);
        std::memcpy(sin_host.data() +
                        static_cast<size_t>(p * head_dim + half_head_dim + j) *
                            elem_sz_,
                    s_src, elem_sz_);
      }
    }

    aclrtMemcpy(cos_table_dev_, table_bytes, cos_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes, sin_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
  }

  int64_t max_seq_len_;

  size_t elem_sz_;

  // Last `cos_sin_cache.data()` uploaded via `UploadCosSinCache()`.  Compared
  // on every call to detect caller-side cache swaps.
  mutable const void* cos_sin_cache_data_ = nullptr;

  // Pre-expanded cos/sin tables on device: `[max_seq_len, head_dim]`.
  void* cos_table_dev_ = nullptr;

  void* sin_table_dev_ = nullptr;

  // Device buffers for gathered `[T, head_dim]` cos/sin.
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
