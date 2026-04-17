#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
// (index=0).
//
// ATB Rope with `rotaryCoeff=2`, `cosFormat=0` expects 5 inputs / 2 outputs:
//   inTensors[0] = query      [T, hiddenSizeQ]
//   inTensors[1] = key        [T, hiddenSizeK]
//   inTensors[2] = cos        [T, headDim]   — pre-gathered per-token cos
//   inTensors[3] = sin        [T, headDim]   — pre-gathered per-token sin
//   inTensors[4] = seqlen     [batch]        — per-batch sequence lengths
//   outTensors[0] = query_out [T, hiddenSizeQ]
//   outTensors[1] = key_out   [T, hiddenSizeK]
//
// This implementation gathers cos/sin from pre-expanded `[max_seq_len, D]`
// tables using `aclnnIndexSelect` on the position indices, then passes the
// gathered `[T, D]` tensors to ATB Rope.  The `seqlen` input is a single
// int32 element equal to T (all tokens treated as one batch).
//
// Restrictions (implementation choices, not ATB API limits):
//   - `rotary_dim` must equal `head_size` (full rotation only).  ATB
//     RopeParam supports `rotaryCoeff=2/4/head_size/head_size_2` per the
//     CANN 8.5 ATB docs; this wrapper plumbs only `rotaryCoeff=2`.
//   - `is_neox_style` must be true.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend, 1>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query, const Tensor key,
           const Tensor cos_sin_cache, int64_t head_size, int64_t rotary_dim,
           bool is_neox_style, Tensor query_out, Tensor key_out)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        rotary_dim, is_neox_style, query_out, key_out) {
    assert(rotary_dim == head_size &&
           "ATB `RotaryEmbedding` requires rotary_dim == head_size");
    assert(is_neox_style &&
           "ATB `RotaryEmbedding` requires neox style (rotaryCoeff=2)");

    const int64_t D = head_size_;
    const size_t elem_sz = cos_sin_cache.element_size();

    max_seq_len_ = cos_sin_cache.size(0);
    size_t table_bytes =
        static_cast<size_t>(max_seq_len_) * static_cast<size_t>(D) * elem_sz;

    // Allocate device buffers for expanded cos/sin tables [max_seq_len, D].
    aclrtMalloc(&cos_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Upload initial cos_sin_cache.  In real inference the cache is loaded
    // once and never mutated, so this one-time upload is sufficient.
    uploadCosSinCache(cos_sin_cache);

    // Cache shapes and metadata.
    const int64_t T = num_tokens_;
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;
    q_2d_shape_ = {T, hiddenQ};
    k_2d_shape_ = {T, hiddenK};
    cos_sin_gathered_shape_ = {T, D};
    seqlen_shape_ = {1};
    acl_dt_ = ascend::toAclDtype(query.dtype());
    elem_size_ = static_cast<uint64_t>(elem_sz);

    // Allocate gathered cos/sin buffers [T, D] — filled by aclnnIndexSelect.
    size_t gathered_bytes = static_cast<size_t>(T * D) * elem_sz;
    aclrtMalloc(&cos_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_dev_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Allocate seqlen buffer: 1 int32 element holding T.
    aclrtMalloc(&seqlen_dev_, sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    int32_t seqlen_val = static_cast<int32_t>(T);
    aclrtMemcpy(seqlen_dev_, sizeof(int32_t), &seqlen_val, sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // IndexSelect descriptor caches: table ptrs stable, positions ptr varies.
    cos_table_cache_ =
        ascend::AclTensorCache({max_seq_len_, D}, acl_dt_, cos_table_dev_);
    sin_table_cache_ =
        ascend::AclTensorCache({max_seq_len_, D}, acl_dt_, sin_table_dev_);
    idx_cache_ = ascend::AclTensorCache({T}, ACL_INT64,
                                        const_cast<void*>(positions.data()));
    cos_out_cache_ = ascend::AclTensorCache({T, D}, acl_dt_, cos_dev_);
    sin_out_cache_ = ascend::AclTensorCache({T, D}, acl_dt_, sin_dev_);

    // Create the ATB Rope operation.
    atb::infer::RopeParam param;
    param.rotaryCoeff = 2;  // Neox half-rotation.
    param.cosFormat = 0;    // Inference mode.
    atb::Status s = atb::CreateOperation(param, &op_);

    assert(s == atb::NO_ERROR && "atb::CreateOperation(Rope) failed");
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

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

  void operator()(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    int64_t T = query.size(0);
    int64_t D = head_size;

    // Compute total hidden sizes for the 2D view expected by ATB Rope.
    // Works for both 2D `[T, N*D]` and 3D `[T, N, D]` input.
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;

    // Re-upload cos/sin tables if the caller passes a different
    // `cos_sin_cache` buffer.  `CacheKey` matches on shape/stride/dtype and
    // ignores data pointers, so a cached operator instance is reused across
    // calls with different cache allocations — see
    // `operator_cache_stale_data` in memory.
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
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_max);

      aclnnIndexSelect(arena.buf, idx_cos_ws_, idx_cos_exec_, stream);
      aclnnIndexSelect(arena.buf, idx_sin_ws_, idx_sin_exec_, stream);
    }

    // Step 2: Copy q->q_out, k->k_out if not in-place.
    size_t elem_sz = query.element_size();

    if (query.data() != query_out.data()) {
      aclrtMemcpyAsync(query_out.data(),
                       static_cast<size_t>(T * hiddenQ) * elem_sz, query.data(),
                       static_cast<size_t>(T * hiddenQ) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    if (key.data() != key_out.data()) {
      aclrtMemcpyAsync(key_out.data(),
                       static_cast<size_t>(T * hiddenK) * elem_sz, key.data(),
                       static_cast<size_t>(T * hiddenK) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    // Step 3: Build ATB VariantPack with 5 inputs + 2 outputs.
    // Inputs: q_out [T, hiddenQ], k_out [T, hiddenK],
    //         cos [T, D], sin [T, D], seqlen [1].
    // Outputs: q_out [T, hiddenQ], k_out [T, hiddenK].
    atb::Context* ctx = ascend::getAtbContext(stream);

    uint64_t q_bytes = static_cast<uint64_t>(T * hiddenQ) * elem_size_;
    uint64_t k_bytes = static_cast<uint64_t>(T * hiddenK) * elem_size_;
    uint64_t gathered_bytes = static_cast<uint64_t>(T * D) * elem_size_;

    atb::Tensor t_q =
        ascend::toAtbTensor(q_2d_shape_, acl_dt_, query_out.data(), q_bytes);
    atb::Tensor t_k =
        ascend::toAtbTensor(k_2d_shape_, acl_dt_, key_out.data(), k_bytes);
    atb::Tensor t_cos = ascend::toAtbTensor(cos_sin_gathered_shape_, acl_dt_,
                                            cos_dev_, gathered_bytes);
    atb::Tensor t_sin = ascend::toAtbTensor(cos_sin_gathered_shape_, acl_dt_,
                                            sin_dev_, gathered_bytes);
    atb::Tensor t_seqlen =
        ascend::toAtbTensor(seqlen_shape_, ACL_INT32, seqlen_dev_,
                            static_cast<uint64_t>(sizeof(int32_t)));

    atb::VariantPack vp;
    vp.inTensors = {t_q, t_k, t_cos, t_sin, t_seqlen};
    vp.outTensors = {t_q, t_k};

    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Setup failed");

    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Execute failed");
  }

 private:
  // D2H copy cos_sin_cache, split into cos/sin, neox-expand, and upload to
  // device.  Called once at construction.
  void uploadCosSinCache(const Tensor cos_sin_cache) const {
    const int64_t D = head_size_;
    const int64_t half_D = D / 2;
    const size_t elem_sz = cos_sin_cache.element_size();
    size_t table_bytes =
        static_cast<size_t>(max_seq_len_) * static_cast<size_t>(D) * elem_sz;

    std::vector<uint8_t> cache_host(table_bytes);
    aclrtMemcpy(cache_host.data(), table_bytes, cos_sin_cache.data(),
                table_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<uint8_t> cos_host(table_bytes);
    std::vector<uint8_t> sin_host(table_bytes);

    for (int64_t p = 0; p < max_seq_len_; ++p) {
      for (int64_t j = 0; j < half_D; ++j) {
        const auto* c_src =
            cache_host.data() + static_cast<size_t>(p * D + j) * elem_sz;
        const auto* s_src = cache_host.data() +
                            static_cast<size_t>(p * D + half_D + j) * elem_sz;

        std::memcpy(cos_host.data() + static_cast<size_t>(p * D + j) * elem_sz,
                    c_src, elem_sz);
        std::memcpy(
            cos_host.data() + static_cast<size_t>(p * D + half_D + j) * elem_sz,
            c_src, elem_sz);
        std::memcpy(sin_host.data() + static_cast<size_t>(p * D + j) * elem_sz,
                    s_src, elem_sz);
        std::memcpy(
            sin_host.data() + static_cast<size_t>(p * D + half_D + j) * elem_sz,
            s_src, elem_sz);
      }
    }

    aclrtMemcpy(cos_table_dev_, table_bytes, cos_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes, sin_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
  }

  atb::Operation* op_ = nullptr;

  // Neox-expanded cos/sin tables on device: [max_seq_len, D].
  void* cos_table_dev_ = nullptr;

  void* sin_table_dev_ = nullptr;

  // Device buffers for gathered [T, D] cos/sin.
  void* cos_dev_ = nullptr;

  void* sin_dev_ = nullptr;

  // Device buffer for seqlen: 1 int32 element holding T.
  void* seqlen_dev_ = nullptr;

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

  // Cached shapes for ATB VariantPack.
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
