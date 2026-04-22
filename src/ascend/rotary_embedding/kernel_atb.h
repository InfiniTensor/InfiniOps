#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_index_select.h"
#include "ascend/atb_common_.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// ATB-based rotary position embedding (implementation index 1).
//
// Wraps ATB `RopeParam` which applies rotary embedding in a single fused
// kernel, eliminating the per-token V2 decomposition in the CANN path
// (index 0).  When `pre_gathered` is true, `cos_sin_cache` is interpreted as
// the already-gathered `[T, head_size * 2]` table (cos half followed by sin
// half, neox or interleave layout chosen upstream); the internal
// `aclnnIndexSelect` step is skipped.
//
// ATB Rope with `rotaryCoeff=2`, `cosFormat=0` expects 5 inputs / 2 outputs:
//   `inTensors[0] = query   [T, hidden_q]`
//   `inTensors[1] = key     [T, hidden_k]`
//   `inTensors[2] = cos     [T, head_dim]`   — pre-gathered per-token cos.
//   `inTensors[3] = sin     [T, head_dim]`   — pre-gathered per-token sin.
//   `inTensors[4] = seqlen  [batch]`         — per-batch sequence lengths.
//   `outTensors[0] = q_out  [T, hidden_q]`
//   `outTensors[1] = k_out  [T, hidden_k]`
//
// This implementation gathers cos/sin from pre-expanded
// `[max_seq_len, head_dim]` tables using `aclnnIndexSelect` on the position
// indices, then passes the gathered `[T, head_dim]` tensors to ATB Rope.
// The `seqlen` input is a single `int32` element equal to `T` (all tokens
// treated as one batch).
//
// Restrictions:
//   - `rotary_dim` must equal `head_size` (full rotation only).  ATB
//     `RopeParam` supports `rotaryCoeff=2/4/head_size/head_size_2` per the
//     CANN 8.5 ATB docs.  This wrapper plumbs:
//       * `rotaryCoeff=2` when `is_neox_style=true`  (half split + cat)
//       * `rotaryCoeff=head_size` when `is_neox_style=false` (interleave)
//     Partial rotary (`rotary_dim < head_size`) is not supported by either
//     the aclnn or ATB fused APIs; callers must pad to `head_size` upstream.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend, 1>
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
                        pre_gathered) {
    assert(rotary_dim == head_size &&
           "Ascend `RotaryEmbedding` (ATB): `rotary_dim` must equal "
           "`head_size` — ATB `RopeParam` does not support partial rotary.");
    assert(has_key_ &&
           "Ascend `RotaryEmbedding` (ATB): `key` is required — ATB "
           "`RopeParam` always rotates Q and K together.");

    const int64_t head_dim = head_size_;
    const size_t elem_sz = cos_sin_cache.element_size();

    max_seq_len_ = cos_sin_cache.size(0);

    const int64_t num_tokens = num_tokens_;
    int64_t hidden_q = static_cast<int64_t>(query.numel()) / num_tokens;
    int64_t hidden_k = static_cast<int64_t>(key->numel()) / num_tokens;
    q_2d_shape_ = {num_tokens, hidden_q};
    k_2d_shape_ = {num_tokens, hidden_k};
    cos_sin_gathered_shape_ = {num_tokens, head_dim};
    seqlen_shape_ = {1};
    acl_dt_ = ascend::ToAclDtype(query.dtype());
    elem_size_ = static_cast<uint64_t>(elem_sz);

    if (!pre_gathered_) {
      size_t table_bytes = static_cast<size_t>(max_seq_len_) *
                           static_cast<size_t>(head_dim) * elem_sz;

      // Allocate device buffers for expanded cos/sin tables
      // `[max_seq_len, head_dim]`.
      aclrtMalloc(&cos_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      aclrtMalloc(&sin_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

      // Upload the initial cos_sin_cache.  `cos_sin_cache_data_` memorizes
      // the source pointer; if the caller later hands in a different buffer,
      // `operator()` re-runs the upload.
      UploadCosSinCache(cos_sin_cache);
      cos_sin_cache_data_ = cos_sin_cache.data();

      // Allocate gathered cos/sin buffers `[T, head_dim]` — filled by
      // `aclnnIndexSelect`.
      size_t gathered_bytes =
          static_cast<size_t>(num_tokens * head_dim) * elem_sz;
      aclrtMalloc(&cos_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      aclrtMalloc(&sin_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

      // IndexSelect descriptor caches: table ptrs stable, positions ptr
      // varies.
      cos_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                                acl_dt_, cos_table_dev_);
      sin_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                                acl_dt_, sin_table_dev_);
      idx_cache_ = ascend::AclTensorCache({num_tokens}, ACL_INT64,
                                          const_cast<void*>(positions.data()));
      cos_out_cache_ =
          ascend::AclTensorCache({num_tokens, head_dim}, acl_dt_, cos_dev_);
      sin_out_cache_ =
          ascend::AclTensorCache({num_tokens, head_dim}, acl_dt_, sin_dev_);
    }

    // Allocate seqlen buffer: 1 `int32` element holding `T`.
    aclrtMalloc(&seqlen_dev_, sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    int32_t seqlen_val = static_cast<int32_t>(num_tokens);
    aclrtMemcpy(seqlen_dev_, sizeof(int32_t), &seqlen_val, sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // Create the ATB Rope operation.  `rotaryCoeff` selects the rotation
    // pattern: `2` for neox (split-then-rotate halves), `head_size` for
    // interleave (pair-wise rotate adjacent elements).
    atb::infer::RopeParam param;
    param.rotaryCoeff = is_neox_style ? 2 : static_cast<int32_t>(head_dim);
    param.cosFormat = 0;  // Inference mode.
    atb::Status s = atb::CreateOperation(param, &op_);

    assert(s == atb::NO_ERROR && "`atb::CreateOperation(Rope)` failed.");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    cos_table_cache_.release();
    sin_table_cache_.release();
    idx_cache_.release();
    cos_out_cache_.release();
    sin_out_cache_.release();

    if (op_) atb::DestroyOperation(op_);
    if (cos_table_dev_) aclrtFree(cos_table_dev_);
    if (sin_table_dev_) aclrtFree(sin_table_dev_);
    if (cos_dev_) aclrtFree(cos_dev_);
    if (sin_dev_) aclrtFree(sin_dev_);
    if (seqlen_dev_) aclrtFree(seqlen_dev_);
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
    // Non-const so `.data()` returns a writable `void*`.
    Tensor q_out = query_out.value_or(query);
    Tensor k_out = key_out.value_or(*key);

    int64_t num_tokens = query.size(0);
    int64_t head_dim = head_size;

    // Compute total hidden sizes for the 2D view expected by ATB Rope.
    // Works for both 2D `[T, N * D]` and 3D `[T, N, D]` input.
    int64_t hidden_q = static_cast<int64_t>(query.numel()) / num_tokens;
    int64_t hidden_k = static_cast<int64_t>(key->numel()) / num_tokens;

    const void* cos_for_rope = nullptr;
    const void* sin_for_rope = nullptr;

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

      cos_for_rope = cos_dev_;
      sin_for_rope = sin_dev_;
    } else {
      // Pre-gathered: caller passes `[2 * T, head_size]` — rows 0..T-1 are
      // expanded cos (neox or interleave per `is_neox_style`), rows T..2T-1
      // are expanded sin (stacked via `torch.cat([cos, sin], dim=0)` in the
      // `apply_rotary_pos_emb` shim).
      const auto* base = static_cast<const uint8_t*>(cos_sin_cache.data());
      cos_for_rope = base;
      sin_for_rope =
          base + static_cast<size_t>(num_tokens * head_dim) * elem_size_;
    }

    // Step 2: Copy q -> q_out, k -> k_out if not in-place.
    size_t elem_sz = query.element_size();

    if (query.data() != q_out.data()) {
      aclrtMemcpyAsync(
          q_out.data(), static_cast<size_t>(num_tokens * hidden_q) * elem_sz,
          query.data(), static_cast<size_t>(num_tokens * hidden_q) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    if (key->data() != k_out.data()) {
      aclrtMemcpyAsync(
          k_out.data(), static_cast<size_t>(num_tokens * hidden_k) * elem_sz,
          key->data(), static_cast<size_t>(num_tokens * hidden_k) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    // Step 3: Build ATB `VariantPack` with 5 inputs + 2 outputs.
    // Inputs: `q_out [T, hidden_q]`, `k_out [T, hidden_k]`,
    //         `cos [T, head_dim]`, `sin [T, head_dim]`, `seqlen [1]`.
    // Outputs: `q_out [T, hidden_q]`, `k_out [T, hidden_k]`.
    atb::Context* ctx = ascend::GetAtbContext(stream);

    uint64_t q_bytes =
        static_cast<uint64_t>(num_tokens * hidden_q) * elem_size_;
    uint64_t k_bytes =
        static_cast<uint64_t>(num_tokens * hidden_k) * elem_size_;
    uint64_t gathered_bytes =
        static_cast<uint64_t>(num_tokens * head_dim) * elem_size_;

    atb::Tensor t_q =
        ascend::ToAtbTensor(q_2d_shape_, acl_dt_, q_out.data(), q_bytes);
    atb::Tensor t_k =
        ascend::ToAtbTensor(k_2d_shape_, acl_dt_, k_out.data(), k_bytes);
    atb::Tensor t_cos =
        ascend::ToAtbTensor(cos_sin_gathered_shape_, acl_dt_,
                            const_cast<void*>(cos_for_rope), gathered_bytes);
    atb::Tensor t_sin =
        ascend::ToAtbTensor(cos_sin_gathered_shape_, acl_dt_,
                            const_cast<void*>(sin_for_rope), gathered_bytes);
    atb::Tensor t_seqlen =
        ascend::ToAtbTensor(seqlen_shape_, ACL_INT32, seqlen_dev_,
                            static_cast<uint64_t>(sizeof(int32_t)));

    atb::VariantPack vp;
    vp.inTensors = {t_q, t_k, t_cos, t_sin, t_seqlen};
    vp.outTensors = {t_q, t_k};

    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope `Setup` failed.");

    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope `Execute` failed.");
  }

 private:
  // D2H copy `cos_sin_cache`, split into cos/sin, expand to
  // `[max_seq_len, head_dim]` in the layout that ATB Rope expects for the
  // chosen `rotaryCoeff`, and upload to device.  Called at construction and
  // on cache-pointer change.
  //
  // For `rotaryCoeff=2` (neox): cos tensor holds the same `half_head_dim`
  // values duplicated front/back —
  // `[c_0 .. c_{half-1}, c_0 .. c_{half-1}]`.
  //
  // For `rotaryCoeff=head_size` (interleave): cos tensor holds each of the
  // `half_head_dim` values repeated pair-wise —
  // `[c_0, c_0, c_1, c_1, .., c_{half-1}, c_{half-1}]`.
  void UploadCosSinCache(const Tensor cos_sin_cache) const {
    const int64_t head_dim = head_size_;
    const int64_t half_head_dim = head_dim / 2;
    const size_t elem_sz = cos_sin_cache.element_size();
    size_t table_bytes = static_cast<size_t>(max_seq_len_) *
                         static_cast<size_t>(head_dim) * elem_sz;

    std::vector<uint8_t> cache_host(table_bytes);
    aclrtMemcpy(cache_host.data(), table_bytes, cos_sin_cache.data(),
                table_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<uint8_t> cos_host(table_bytes);
    std::vector<uint8_t> sin_host(table_bytes);

    for (int64_t p = 0; p < max_seq_len_; ++p) {
      for (int64_t j = 0; j < half_head_dim; ++j) {
        const auto* c_src =
            cache_host.data() + static_cast<size_t>(p * head_dim + j) * elem_sz;
        const auto* s_src =
            cache_host.data() +
            static_cast<size_t>(p * head_dim + half_head_dim + j) * elem_sz;

        if (is_neox_style_) {
          // Neox layout: `[c_j ... , c_j ...]` front/back duplication.
          std::memcpy(
              cos_host.data() + static_cast<size_t>(p * head_dim + j) * elem_sz,
              c_src, elem_sz);
          std::memcpy(cos_host.data() + static_cast<size_t>(p * head_dim +
                                                            half_head_dim + j) *
                                            elem_sz,
                      c_src, elem_sz);
          std::memcpy(
              sin_host.data() + static_cast<size_t>(p * head_dim + j) * elem_sz,
              s_src, elem_sz);
          std::memcpy(sin_host.data() + static_cast<size_t>(p * head_dim +
                                                            half_head_dim + j) *
                                            elem_sz,
                      s_src, elem_sz);
        } else {
          // Interleave layout: each value repeated pair-wise.
          std::memcpy(cos_host.data() +
                          static_cast<size_t>(p * head_dim + 2 * j) * elem_sz,
                      c_src, elem_sz);
          std::memcpy(
              cos_host.data() +
                  static_cast<size_t>(p * head_dim + 2 * j + 1) * elem_sz,
              c_src, elem_sz);
          std::memcpy(sin_host.data() +
                          static_cast<size_t>(p * head_dim + 2 * j) * elem_sz,
                      s_src, elem_sz);
          std::memcpy(
              sin_host.data() +
                  static_cast<size_t>(p * head_dim + 2 * j + 1) * elem_sz,
              s_src, elem_sz);
        }
      }
    }

    aclrtMemcpy(cos_table_dev_, table_bytes, cos_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes, sin_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
  }

  atb::Operation* op_ = nullptr;

  // Neox-expanded cos/sin tables on device: `[max_seq_len, head_dim]`.
  void* cos_table_dev_ = nullptr;

  void* sin_table_dev_ = nullptr;

  // Device buffers for gathered `[T, head_dim]` cos/sin.
  void* cos_dev_ = nullptr;

  void* sin_dev_ = nullptr;

  // Device buffer for `seqlen`: 1 `int32` element holding `T`.
  void* seqlen_dev_ = nullptr;

  // Last `cos_sin_cache.data()` uploaded via `UploadCosSinCache()`.  Compared
  // on every call to detect caller-side cache swaps.
  mutable const void* cos_sin_cache_data_ = nullptr;

  // IndexSelect descriptor caches.
  mutable ascend::AclTensorCache cos_table_cache_;

  mutable ascend::AclTensorCache sin_table_cache_;

  mutable ascend::AclTensorCache idx_cache_;

  mutable ascend::AclTensorCache cos_out_cache_;

  mutable ascend::AclTensorCache sin_out_cache_;

  // Cached IndexSelect executors.
  mutable aclOpExecutor* idx_cos_exec_ = nullptr;

  mutable uint64_t idx_cos_ws_ = 0;

  mutable aclOpExecutor* idx_sin_exec_ = nullptr;

  mutable uint64_t idx_sin_ws_ = 0;

  // Cached shapes for ATB `VariantPack`.
  std::vector<int64_t> q_2d_shape_;

  std::vector<int64_t> k_2d_shape_;

  std::vector<int64_t> cos_sin_gathered_shape_;

  std::vector<int64_t> seqlen_shape_;

  aclDataType acl_dt_ = ACL_DT_UNDEFINED;

  uint64_t elem_size_ = 0;

  int64_t max_seq_len_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_
