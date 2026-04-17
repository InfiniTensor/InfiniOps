#ifndef INFINI_OPS_ASCEND_PAGED_ATTENTION_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_PAGED_ATTENTION_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "ascend/atb_common_.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "base/paged_attention.h"
#include "operator.h"

namespace infini::ops {

// ATB-based paged decode attention (implementation index 0).
//
// Wraps ATB `PagedAttentionParam` with the default `inputLayout`
// (`TYPE_BSND`).  For decode (single token per request) the S
// dimension is implicitly 1, so query and output use 3D shape
// [batch, num_heads, head_size] matching vLLM's convention.
//
// ATB internally constructs `aclIntArray*` from the `hostData` field
// of `block_table` and `context_lens` tensors.  By default the operator
// performs synchronous D2H copies for these two small tensors each call.
// When the caller provides `seq_lens_host` and `block_table_host` (CPU
// pinned tensors), the D2H copies are skipped entirely — enabling full
// NPUGraph capture of the decode attention path.
//
// ATB VariantPack layout (BSND with S=1):
//   inTensors[0] = query         [B, N, D]
//   inTensors[1] = key_cache     [num_blocks, block_size, Nkv, D]
//   inTensors[2] = value_cache   [num_blocks, block_size, Nkv, D]
//   inTensors[3] = block_table   [B, max_num_blocks]  (device + host)
//   inTensors[4] = context_lens  [B]  (int32)         (device + host)
//   outTensors[0] = output       [B, N, D]
template <>
class Operator<PagedAttention, Device::Type::kAscend, 0>
    : public PagedAttention {
 public:
  Operator(const Tensor query, const Tensor key_cache, const Tensor value_cache,
           const Tensor seq_lens, const Tensor block_table, int64_t num_heads,
           int64_t num_kv_heads, int64_t head_size, double scale,
           int64_t block_size, Tensor output,
           std::optional<Tensor> seq_lens_host = std::nullopt,
           std::optional<Tensor> block_table_host = std::nullopt)
      : PagedAttention(query, key_cache, value_cache, seq_lens, block_table,
                       num_heads, num_kv_heads, head_size, scale, block_size,
                       output, seq_lens_host, block_table_host) {
    int64_t B = static_cast<int64_t>(batch_size_);
    int64_t N = num_heads_;
    int64_t Nkv = num_kv_heads_;
    int64_t D = head_size_;

    // Query/output shapes: 3D [B, N, D] (BSND with S=1 for decode).
    query_tnd_shape_ = {B, N, D};
    output_tnd_shape_ = {B, N, D};

    // KV cache shapes.
    int64_t num_blocks = static_cast<int64_t>(key_cache.size(0));
    int64_t bs = static_cast<int64_t>(key_cache.size(1));
    kv_cache_shape_ = {num_blocks, bs, Nkv, D};

    // Block table and context lens shapes.
    int64_t max_blocks = static_cast<int64_t>(block_table.size(1));
    block_table_shape_ = {B, max_blocks};
    context_lens_shape_ = {B};

    // ACL data types.
    acl_dt_ = ascend::toAclDtype(query.dtype());
    bt_dt_ = ascend::toAclDtype(block_table.dtype());
    sl_dt_ = ascend::toAclDtype(seq_lens.dtype());

    // Element sizes for `dataSize` computation.
    elem_size_ = query.element_size();
    bt_elem_size_ = block_table.element_size();
    sl_elem_size_ = seq_lens.element_size();

    // Pre-allocate pinned host buffers for D2H copies.
    // ATB PA reads `hostData` from block_table and context_lens to
    // construct internal `aclIntArray*` parameters.
    // When caller provides host tensors, skip allocation — the caller's
    // pinned buffers will be used directly in `operator()`.
    bt_host_bytes_ = static_cast<uint64_t>(B * max_blocks) * bt_elem_size_;
    sl_host_bytes_ = static_cast<uint64_t>(B) * sl_elem_size_;

    if (!has_block_table_host_) {
      bt_host_ = std::malloc(bt_host_bytes_);
      assert(bt_host_ && "Host buffer allocation for `block_table` failed");
    }

    if (!has_seq_lens_host_) {
      sl_host_ = std::malloc(sl_host_bytes_);
      assert(sl_host_ && "Host buffer allocation for `seq_lens` failed");
    }

    // Create the ATB operation (reused across calls).
    atb::infer::PagedAttentionParam param;
    param.headNum = static_cast<int32_t>(N);
    param.kvHeadNum = static_cast<int32_t>(Nkv);
    param.qkScale = static_cast<float>(scale_);

    atb::Status s = atb::CreateOperation(param, &op_);
    assert(s == atb::NO_ERROR && "atb::CreateOperation(PagedAttention) failed");
  }

  ~Operator() {
    // Host memory is always safe to free.
    if (!has_block_table_host_) {
      std::free(bt_host_);
    }

    if (!has_seq_lens_host_) {
      std::free(sl_host_);
    }

    if (!ascend::isAclRuntimeAlive()) return;

    if (op_) {
      atb::DestroyOperation(op_);
    }
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor query, const Tensor key_cache,
                  const Tensor value_cache, const Tensor seq_lens,
                  const Tensor block_table, int64_t num_heads,
                  int64_t num_kv_heads, int64_t head_size, double scale,
                  int64_t block_size, Tensor output,
                  std::optional<Tensor> seq_lens_host,
                  std::optional<Tensor> block_table_host) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    atb::Context* ctx = ascend::getAtbContext(stream);

    // Use caller-provided host data or perform synchronous D2H copy.
    // ATB reads `hostData` to construct internal `aclIntArray*`.
    void* bt_host_ptr = bt_host_;
    void* sl_host_ptr = sl_host_;

    if (block_table_host.has_value()) {
      bt_host_ptr = const_cast<void*>(block_table_host.value().data());
    } else {
      aclrtMemcpy(bt_host_, bt_host_bytes_, block_table.data(), bt_host_bytes_,
                  ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (seq_lens_host.has_value()) {
      sl_host_ptr = const_cast<void*>(seq_lens_host.value().data());
    } else {
      aclrtMemcpy(sl_host_, sl_host_bytes_, seq_lens.data(), sl_host_bytes_,
                  ACL_MEMCPY_DEVICE_TO_HOST);
    }

    atb::VariantPack vp = buildVariantPack(
        const_cast<void*>(query.data()), const_cast<void*>(key_cache.data()),
        const_cast<void*>(value_cache.data()),
        const_cast<void*>(block_table.data()),
        const_cast<void*>(seq_lens.data()), output.data(), bt_host_ptr,
        sl_host_ptr);

    // Setup computes workspace requirements and binds tensor descriptors.
    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);
    assert(s == atb::NO_ERROR &&
           "atb::Operation::Setup(PagedAttention) failed");

    // Allocate workspace via the shared pool.
    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::workspacePool().ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);
    assert(s == atb::NO_ERROR &&
           "atb::Operation::Execute(PagedAttention) failed");
  }

 private:
  // Build the ATB VariantPack.
  //
  // Query and output are 3D [B, N, D] (BSND with S=1 for decode).
  // Block table and context lens carry both `deviceData` and
  // `hostData` because ATB reads the host copy to build internal
  // `aclIntArray*` parameters.
  atb::VariantPack buildVariantPack(void* query_data, void* key_cache_data,
                                    void* value_cache_data,
                                    void* block_table_data, void* seq_lens_data,
                                    void* output_data, void* bt_host_ptr,
                                    void* sl_host_ptr) const {
    int64_t B = query_tnd_shape_[0];
    int64_t N = query_tnd_shape_[1];
    int64_t D = query_tnd_shape_[2];

    // Query [B, N, D] — 3D (BSND with S=1).
    uint64_t q_bytes = static_cast<uint64_t>(B * N * D) * elem_size_;
    atb::Tensor t_query =
        ascend::toAtbTensor(query_tnd_shape_, acl_dt_, query_data, q_bytes);

    // KV caches [num_blocks, block_size, Nkv, D].
    int64_t nb = kv_cache_shape_[0];
    int64_t bs = kv_cache_shape_[1];
    int64_t Nkv = kv_cache_shape_[2];
    uint64_t kv_bytes = static_cast<uint64_t>(nb * bs * Nkv * D) * elem_size_;
    atb::Tensor t_key_cache =
        ascend::toAtbTensor(kv_cache_shape_, acl_dt_, key_cache_data, kv_bytes);
    atb::Tensor t_value_cache = ascend::toAtbTensor(kv_cache_shape_, acl_dt_,
                                                    value_cache_data, kv_bytes);

    // Block table [B, max_blocks] — with hostData for `aclIntArray*`.
    atb::Tensor t_block_table = ascend::toAtbTensor(
        block_table_shape_, bt_dt_, block_table_data, bt_host_bytes_);
    t_block_table.hostData = bt_host_ptr;

    // Context lens [B] — with hostData for `aclIntArray*`.
    atb::Tensor t_context_lens = ascend::toAtbTensor(
        context_lens_shape_, sl_dt_, seq_lens_data, sl_host_bytes_);
    t_context_lens.hostData = sl_host_ptr;

    // Output [B, N, D] — 3D (BSND with S=1).
    atb::Tensor t_output =
        ascend::toAtbTensor(output_tnd_shape_, acl_dt_, output_data, q_bytes);

    atb::VariantPack vp;
    vp.inTensors = {t_query, t_key_cache, t_value_cache, t_block_table,
                    t_context_lens};
    vp.outTensors = {t_output};

    return vp;
  }

  atb::Operation* op_ = nullptr;

  std::vector<int64_t> query_tnd_shape_;

  std::vector<int64_t> output_tnd_shape_;

  std::vector<int64_t> kv_cache_shape_;

  std::vector<int64_t> block_table_shape_;

  std::vector<int64_t> context_lens_shape_;

  aclDataType acl_dt_ = ACL_DT_UNDEFINED;

  aclDataType bt_dt_ = ACL_DT_UNDEFINED;

  aclDataType sl_dt_ = ACL_DT_UNDEFINED;

  uint64_t elem_size_ = 0;

  uint64_t bt_elem_size_ = 0;

  uint64_t sl_elem_size_ = 0;

  // Host-side buffers for ATB's internal `aclIntArray*` construction.
  void* bt_host_ = nullptr;

  void* sl_host_ = nullptr;

  uint64_t bt_host_bytes_ = 0;

  uint64_t sl_host_bytes_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_PAGED_ATTENTION_KERNEL_ATB_H_
