#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "ascend/common.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "ascend/atb_common_.h"
#include "ascend/rotary_embedding/registry.h"
#include "ascend/workspace_pool_.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// ATB-based rotary position embedding (implementation index 1).
//
// Wraps ATB `RopeParam` which applies rotary embedding in a single fused
// kernel.  ATB Rope handles position gathering internally, eliminating
// the 2x `aclnnIndexSelect` calls that produce ~62k GatherV3+Slice
// kernels per inference step in the CANN path (index=0).
//
// ATB Rope expects 5 inputs and 2 outputs:
//   inTensors[0] = query        [num_tokens, hiddenSizeQ]
//   inTensors[1] = key          [num_tokens, hiddenSizeK]
//   inTensors[2] = cos_table    [max_seq_len, headDim]
//   inTensors[3] = sin_table    [max_seq_len, headDim]
//   inTensors[4] = seq_len      [num_tokens] (int32, position indices)
//   outTensors[0] = query_out   [num_tokens, hiddenSizeQ]
//   outTensors[1] = key_out     [num_tokens, hiddenSizeK]
//
// The constructor splits the cos_sin_cache into separate cos/sin
// device tables [max_seq_len, headDim] with neox expansion.
//
// Restrictions:
//   - rotary_dim must equal head_size (full rotation only).
//   - is_neox_style must be true (rotaryCoeff=2).
//   - fp16 only (ATB inference constraint).
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

    const int64_t max_seq_len = cos_sin_cache.size(0);
    const int64_t D = head_size_;
    const int64_t half_D = D / 2;
    const size_t elem_sz = cos_sin_cache.element_size();

    // One-time: D2H copy cos_sin_cache, split into cos/sin, upload.
    // cos_sin_cache layout per row: [c0..c_{hD-1}, s0..s_{hD-1}].
    size_t row_bytes = static_cast<size_t>(D) * elem_sz;
    size_t table_bytes = static_cast<size_t>(max_seq_len) * row_bytes;

    std::vector<uint8_t> cache_host(table_bytes);
    aclrtMemcpy(cache_host.data(), table_bytes, cos_sin_cache.data(),
                table_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // ATB Rope with rotaryCoeff=2 expects cos/sin of shape [S, D].
    // Neox-style expansion: [c0..c_{hD-1}, c0..c_{hD-1}].
    std::vector<uint8_t> cos_host(table_bytes);
    std::vector<uint8_t> sin_host(table_bytes);

    for (int64_t p = 0; p < max_seq_len; ++p) {
      for (int64_t j = 0; j < half_D; ++j) {
        const auto* c_src =
            cache_host.data() + static_cast<size_t>(p * D + j) * elem_sz;
        const auto* s_src =
            cache_host.data() +
            static_cast<size_t>(p * D + half_D + j) * elem_sz;

        std::memcpy(
            cos_host.data() + static_cast<size_t>(p * D + j) * elem_sz, c_src,
            elem_sz);
        std::memcpy(
            cos_host.data() +
                static_cast<size_t>(p * D + half_D + j) * elem_sz,
            c_src, elem_sz);
        std::memcpy(
            sin_host.data() + static_cast<size_t>(p * D + j) * elem_sz, s_src,
            elem_sz);
        std::memcpy(
            sin_host.data() +
                static_cast<size_t>(p * D + half_D + j) * elem_sz,
            s_src, elem_sz);
      }
    }

    // Upload expanded tables to device (persistent, reused across calls).
    aclrtMalloc(&cos_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_table_dev_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(cos_table_dev_, table_bytes, cos_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes, sin_host.data(), table_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE);

    // Cache shapes and metadata.
    // Query/key may be 2D [T, N*D] or 3D [T, N, D].  Derive the total hidden
    // size directly from the tensor to handle both layouts.
    const int64_t T = num_tokens_;
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;
    q_2d_shape_ = {T, hiddenQ};
    k_2d_shape_ = {T, hiddenK};
    cos_sin_table_shape_ = {max_seq_len, D};
    pos_shape_ = {T};
    acl_dt_ = ascend::toAclDtype(query.dtype());
    elem_size_ = static_cast<uint64_t>(elem_sz);
    max_seq_len_ = max_seq_len;

    // Create the ATB Rope operation.
    atb::infer::RopeParam param;
    param.rotaryCoeff = 2;  // Neox half-rotation.
    param.cosFormat = 0;    // Inference mode.
    atb::Status s = atb::CreateOperation(param, &op_);

    assert(s == atb::NO_ERROR && "atb::CreateOperation(Rope) failed");
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;
    if (op_) atb::DestroyOperation(op_);
    if (cos_table_dev_) aclrtFree(cos_table_dev_);
    if (sin_table_dev_) aclrtFree(sin_table_dev_);
    if (pos_buf_dev_) aclrtFree(pos_buf_dev_);
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

    // Query/key may be 2D [T, N*D] or 3D [T, N, D].  Compute total hidden
    // sizes from the tensor element count to handle both layouts.
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;

    // Copy q→q_out, k→k_out if not in-place.
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

    // Provide int32 positions to ATB.  When the caller pre-casts to int32
    // (required for NPU graph capture), a device-to-device copy suffices.
    // The D2H+sync fallback remains for standalone tests with int64 positions.
    size_t pos32_bytes = static_cast<size_t>(T) * sizeof(int32_t);

    if (pos32_bytes > pos_buf_size_) {
      if (pos_buf_dev_) aclrtFree(pos_buf_dev_);
      aclrtMalloc(&pos_buf_dev_, pos32_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      pos_buf_size_ = pos32_bytes;
    }

    if (positions.element_size() == sizeof(int32_t)) {
      // Already int32 — async D2D copy, graph-compatible.
      aclrtMemcpyAsync(pos_buf_dev_, pos32_bytes, positions.data(),
                       pos32_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    } else {
      // int64 fallback — D2H, CPU cast, H2D (not graph-compatible).
      std::vector<int64_t> pos_i64(static_cast<size_t>(T));
      aclrtMemcpyAsync(pos_i64.data(),
                       static_cast<size_t>(T) * sizeof(int64_t),
                       positions.data(),
                       static_cast<size_t>(T) * sizeof(int64_t),
                       ACL_MEMCPY_DEVICE_TO_HOST, stream);
      aclrtSynchronizeStream(stream);

      std::vector<int32_t> pos_i32(static_cast<size_t>(T));

      for (int64_t i = 0; i < T; ++i) {
        pos_i32[static_cast<size_t>(i)] =
            static_cast<int32_t>(pos_i64[static_cast<size_t>(i)]);
      }

      aclrtMemcpyAsync(pos_buf_dev_, pos32_bytes, pos_i32.data(), pos32_bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }

    // Build ATB VariantPack with 5 inputs + 2 outputs.
    atb::Context* ctx = ascend::getAtbContext(stream);

    uint64_t q_bytes = static_cast<uint64_t>(T * hiddenQ) * elem_size_;
    uint64_t k_bytes = static_cast<uint64_t>(T * hiddenK) * elem_size_;
    uint64_t table_bytes =
        static_cast<uint64_t>(max_seq_len_ * D) * elem_size_;

    atb::Tensor t_q =
        ascend::toAtbTensor(q_2d_shape_, acl_dt_, query_out.data(), q_bytes);
    atb::Tensor t_k =
        ascend::toAtbTensor(k_2d_shape_, acl_dt_, key_out.data(), k_bytes);
    atb::Tensor t_cos = ascend::toAtbTensor(cos_sin_table_shape_, acl_dt_,
                                            cos_table_dev_, table_bytes);
    atb::Tensor t_sin = ascend::toAtbTensor(cos_sin_table_shape_, acl_dt_,
                                            sin_table_dev_, table_bytes);
    atb::Tensor t_pos = ascend::toAtbTensor(pos_shape_, ACL_INT32,
                                            pos_buf_dev_, pos32_bytes);

    atb::VariantPack vp;
    vp.inTensors = {t_q, t_k, t_cos, t_sin, t_pos};
    vp.outTensors = {t_q, t_k};

    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Setup failed");

    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::workspacePool().ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Execute failed");
  }

 private:
  atb::Operation* op_ = nullptr;

  // Neox-expanded cos/sin tables on device: [max_seq_len, D].
  void* cos_table_dev_ = nullptr;

  void* sin_table_dev_ = nullptr;

  // Reusable int32 positions buffer on device.
  mutable void* pos_buf_dev_ = nullptr;

  mutable size_t pos_buf_size_ = 0;

  // Cached shapes for ATB VariantPack.
  std::vector<int64_t> q_2d_shape_;

  std::vector<int64_t> k_2d_shape_;

  std::vector<int64_t> cos_sin_table_shape_;

  std::vector<int64_t> pos_shape_;

  aclDataType acl_dt_ = ACL_DT_UNDEFINED;

  uint64_t elem_size_ = 0;

  int64_t max_seq_len_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_ATB_H_
