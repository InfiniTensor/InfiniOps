#ifndef INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/flash_attention.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Extract cu_seqlens differences to a host aclIntArray.
// cu_seqlens = [0, s1, s1+s2, ...] -> per_seq_lens = [s1, s2, ...].
// Used by paged decode (actualSeqLengthsKv = per-sequence KV lengths).
//
// When cu_seqlens is a CPU tensor (device type kCpu), the data pointer is
// already on the host and can be read directly — no D2H sync needed.
inline aclIntArray* extractSeqLengths(const Tensor& cu_seqlens,
                                      aclrtStream stream) {
  auto n = cu_seqlens.numel();

  const int64_t* cu_host_ptr = nullptr;
  std::vector<int64_t> cu_host_buf;

  if (cu_seqlens.device().type() == Device::Type::kCpu) {
    cu_host_ptr = static_cast<const int64_t*>(cu_seqlens.data());
  } else {
    cu_host_buf.resize(n);
    aclrtMemcpyAsync(cu_host_buf.data(), n * sizeof(int64_t), cu_seqlens.data(),
                     n * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);
    cu_host_ptr = cu_host_buf.data();
  }

  std::vector<int64_t> lengths(n - 1);
  for (size_t i = 0; i < lengths.size(); ++i) {
    lengths[i] = cu_host_ptr[i + 1] - cu_host_ptr[i];
  }

  return aclCreateIntArray(lengths.data(),
                           static_cast<int64_t>(lengths.size()));
}

// Extract cumulative end positions from cu_seqlens to a host aclIntArray.
// cu_seqlens = [0, s1, s1+s2, ...] -> cum_lens = [s1, s1+s2, ...].
// FIA V4 TND varlen uses cumulative end positions, matching the vllm-ascend
// convention for npu_fused_infer_attention_score actual_seq_lengths.
//
// When cu_seqlens is a CPU tensor, reads directly from host memory.
inline aclIntArray* cumSeqLengths(const Tensor& cu_seqlens,
                                  aclrtStream stream) {
  auto n = cu_seqlens.numel();

  const int64_t* cu_host_ptr = nullptr;
  std::vector<int64_t> cu_host_buf;

  if (cu_seqlens.device().type() == Device::Type::kCpu) {
    cu_host_ptr = static_cast<const int64_t*>(cu_seqlens.data());
  } else {
    cu_host_buf.resize(n);
    aclrtMemcpyAsync(cu_host_buf.data(), n * sizeof(int64_t), cu_seqlens.data(),
                     n * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);
    cu_host_ptr = cu_host_buf.data();
  }

  // Skip the leading 0; return [s1, s1+s2, ...].
  return aclCreateIntArray(cu_host_ptr + 1, static_cast<int64_t>(n - 1));
}

// Allocate a 2048x2048 lower-triangular UINT8 causal mask on device.
// Required for `sparseMode` >= 2.
inline aclTensor* makeCausalMask(void** mask_buf, aclrtStream stream) {
  constexpr int64_t kMaskDim = 2048;
  const int64_t mask_elems = kMaskDim * kMaskDim;
  const size_t mask_bytes = static_cast<size_t>(mask_elems);  // uint8_t

  aclrtMalloc(mask_buf, mask_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

  std::vector<uint8_t> host_mask(mask_elems);
  for (int64_t r = 0; r < kMaskDim; ++r) {
    for (int64_t c = 0; c < kMaskDim; ++c) {
      // 1 = masked out (upper triangle); 0 = attend (lower triangle).
      host_mask[r * kMaskDim + c] = (c > r) ? 1 : 0;
    }
  }
  aclrtMemcpyAsync(*mask_buf, mask_bytes, host_mask.data(), mask_bytes,
                   ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);

  std::vector<int64_t> mask_shape = {kMaskDim, kMaskDim};
  std::vector<int64_t> mask_strides = {kMaskDim, 1};
  std::vector<int64_t> mask_storage = {mask_elems};
  return aclCreateTensor(mask_shape.data(), 2, ACL_UINT8, mask_strides.data(),
                         0, ACL_FORMAT_ND, mask_storage.data(), 1, *mask_buf);
}

}  // namespace detail

template <>
class Operator<FlashAttention, Device::Type::kAscend> : public FlashAttention {
 public:
  Operator(const Tensor query, const Tensor key, const Tensor value,
           std::optional<Tensor> cu_seqlens_q,
           std::optional<Tensor> cu_seqlens_kv,
           std::optional<Tensor> block_table, int64_t num_heads,
           int64_t num_kv_heads, int64_t head_size, double scale, bool causal,
           int64_t window_left, int64_t window_right, int64_t block_size,
           Tensor output)
      : FlashAttention(query, key, value, cu_seqlens_q, cu_seqlens_kv,
                       block_table, num_heads, num_kv_heads, head_size, scale,
                       causal, window_left, window_right, block_size, output) {
    paged_ = block_table.has_value() && block_size > 0;
    aclDataType acl_dt = ascend::toAclDtype(query.dtype());

    if (!paged_) {
      // Prefill: cache Q and output (TND layout).
      prefill_q_cache_ = ascend::AclTensorCache(query);
      prefill_out_cache_ = ascend::AclTensorCache(output);

      // Pre-compute causal mask once (sparse_mode >= 2).
      if (causal) {
        int64_t sm = (window_left >= 0) ? 4 : 3;
        if (sm >= 2) {
          causal_mask_ = detail::makeCausalMask(&causal_mask_buf_, nullptr);
        }
      }
    } else {
      // Decode: cache Q/output (BNSD), block_table.
      const int64_t N = query.size(1);
      const int64_t D = query.size(2);
      const int64_t B = query.size(0);

      decode_q_cache_ = ascend::AclTensorCache({B, N, 1, D}, acl_dt,
                                               const_cast<void*>(query.data()));
      decode_out_cache_ =
          ascend::AclTensorCache({B, N, 1, D}, acl_dt, output.data());
      block_table_cache_ = ascend::AclTensorCache(block_table.value());

      // Pre-compute KV reshape metadata.
      const int64_t nb = key.size(0);
      const int64_t bsz = key.size(1);
      const int64_t NkvD = key.size(2) * key.size(3);
      kv_shape_ = {nb, bsz, NkvD};
      kv_strides_ = {bsz * NkvD, NkvD, 1};
      kv_storage_shape_ = {nb * bsz * NkvD};
      kv_acl_dt_ = acl_dt;
    }
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    if (causal_mask_) aclDestroyTensor(causal_mask_);
    if (causal_mask_buf_) aclrtFree(causal_mask_buf_);
  }

  void operator()(const Tensor query, const Tensor key, const Tensor value,
                  std::optional<Tensor> cu_seqlens_q,
                  std::optional<Tensor> cu_seqlens_kv,
                  std::optional<Tensor> block_table, int64_t num_heads,
                  int64_t num_kv_heads, int64_t head_size, double scale,
                  bool causal, int64_t window_left, int64_t window_right,
                  int64_t block_size, Tensor output) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    const bool paged = paged_;

    int64_t sparse_mode;
    int64_t pre_tokens = 2147483647;
    int64_t next_tokens = 2147483647;
    if (causal) {
      if (window_left >= 0) {
        sparse_mode = 4;
        pre_tokens = window_left;
        next_tokens = 0;
      } else {
        sparse_mode = 3;
        next_tokens = 0;
      }
    } else {
      sparse_mode = 0;
      if (window_left >= 0) pre_tokens = window_left;
      if (window_right >= 0) next_tokens = window_right;
    }

    if (!paged) {
      // --- Prefill ---
      int64_t T = query.size(0);

      // cumSeqLengths / extractSeqLengths automatically skip D2H when
      // cu_seqlens is a CPU tensor (see detail:: helpers above).
      aclIntArray* seq_q =
          cu_seqlens_q.has_value()
              ? detail::cumSeqLengths(cu_seqlens_q.value(), stream)
              : aclCreateIntArray(&T, 1);
      aclIntArray* seq_kv =
          cu_seqlens_kv.has_value()
              ? detail::cumSeqLengths(cu_seqlens_kv.value(), stream)
              : aclCreateIntArray(&T, 1);

      aclTensor* t_q = prefill_q_cache_.get(const_cast<void*>(query.data()));
      // K/V descriptors go into TensorList which takes ownership — must be
      // per-call (cannot cache).
      aclTensor* t_k = ascend::buildAclTensor(key);
      aclTensor* t_v = ascend::buildAclTensor(value);
      aclTensor* t_out = prefill_out_cache_.get(output.data());

      const aclTensor* k_arr[] = {t_k};
      const aclTensor* v_arr[] = {t_v};
      aclTensorList* key_list = aclCreateTensorList(k_arr, 1);
      aclTensorList* val_list = aclCreateTensorList(v_arr, 1);

      uint64_t ws_needed = 0;
      aclOpExecutor* executor = nullptr;
      aclError gws = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
          t_q, key_list, val_list,
          nullptr,       // pseShift
          causal_mask_,  // attenMask (pre-computed, or nullptr)
          seq_q,         // actualSeqLengths
          seq_kv,        // actualSeqLengthsKv
          nullptr, nullptr, nullptr, nullptr,
          nullptr,           // deqScale1..quantOffset2
          nullptr, nullptr,  // antiquantScale, antiquantOffset
          nullptr,           // blockTable
          nullptr, nullptr,  // queryPaddingSize, kvPaddingSize
          nullptr, nullptr, nullptr,
          nullptr,  // key/value antiquant scale/offset
          nullptr, nullptr,
          nullptr,  // keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen
          nullptr, nullptr,
          nullptr,           // queryRope, keyRope, keyRopeAntiquantScale
          nullptr, nullptr,  // dequantScaleQuery, learnableSink
          num_heads, scale, pre_tokens, next_tokens, const_cast<char*>("TND"),
          num_kv_heads, sparse_mode,
          0,         // innerPrecise
          0,         // blockSize (unused for prefill)
          0, false,  // antiquantMode, softmaxLseFlag
          0, 0, 0,   // keyAntiquantMode, valueAntiquantMode, queryQuantMode
          t_out, nullptr, &ws_needed, &executor);
      assert(
          gws == ACL_SUCCESS &&
          "aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed (prefill)");

      auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
      aclError ret = aclnnFusedInferAttentionScoreV4(arena.buf, ws_needed,
                                                     executor, stream);
      assert(ret == ACL_SUCCESS &&
             "aclnnFusedInferAttentionScoreV4 failed (prefill)");

      // t_q and t_out are owned by caches — do NOT destroy.
      // t_k and t_v are owned by TensorLists.
      aclDestroyTensorList(key_list);
      aclDestroyTensorList(val_list);
      aclDestroyIntArray(seq_q);
      aclDestroyIntArray(seq_kv);
      return;
    }

    // --- Paged decode ---
    assert(cu_seqlens_kv.has_value() &&
           "`FlashAttention` paged decode requires `cu_seqlens_kv`");

    aclTensor* t_query = decode_q_cache_.get(const_cast<void*>(query.data()));
    aclTensor* t_output = decode_out_cache_.get(output.data());

    // K/V descriptors go into TensorList which takes ownership — must be
    // per-call.  Use pre-computed metadata to avoid heap allocs.
    aclTensor* t_key = aclCreateTensor(
        kv_shape_.data(), static_cast<int64_t>(kv_shape_.size()), kv_acl_dt_,
        kv_strides_.data(), 0, ACL_FORMAT_ND, kv_storage_shape_.data(),
        static_cast<int64_t>(kv_storage_shape_.size()),
        const_cast<void*>(key.data()));
    aclTensor* t_value = aclCreateTensor(
        kv_shape_.data(), static_cast<int64_t>(kv_shape_.size()), kv_acl_dt_,
        kv_strides_.data(), 0, ACL_FORMAT_ND, kv_storage_shape_.data(),
        static_cast<int64_t>(kv_storage_shape_.size()),
        const_cast<void*>(value.data()));

    // extractSeqLengths skips D2H when cu_seqlens_kv is a CPU tensor.
    aclIntArray* seq_kv =
        detail::extractSeqLengths(cu_seqlens_kv.value(), stream);
    aclTensor* t_block_table =
        block_table_cache_.get(const_cast<void*>(block_table.value().data()));

    const aclTensor* k_arr[] = {t_key};
    const aclTensor* v_arr[] = {t_value};
    aclTensorList* key_list = aclCreateTensorList(k_arr, 1);
    aclTensorList* val_list = aclCreateTensorList(v_arr, 1);

    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    aclError gws = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_query, key_list, val_list,
        nullptr,  // pseShift
        nullptr,  // attenMask (sparseMode ignored for Q_S=1)
        nullptr,  // actualSeqLengths (ignored for Q_S=1)
        seq_kv,   // actualSeqLengthsKv (mandatory for paged)
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        t_block_table,  // blockTable
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, num_heads, scale,
        static_cast<int64_t>(2147483647), static_cast<int64_t>(2147483647),
        const_cast<char*>("BNSD"), num_kv_heads,
        0,           // sparseMode=0 (ignored for Q_S=1)
        0,           // innerPrecise
        block_size,  // blockSize
        0, false,    // antiquantMode, softmaxLseFlag
        0, 0, 0,     // keyAntiquantMode, valueAntiquantMode, queryQuantMode
        t_output, nullptr, &ws_needed, &executor);
    assert(gws == ACL_SUCCESS &&
           "aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed (decode)");

    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclError ret =
        aclnnFusedInferAttentionScoreV4(arena.buf, ws_needed, executor, stream);
    assert(ret == ACL_SUCCESS &&
           "aclnnFusedInferAttentionScoreV4 failed (decode)");

    // t_query, t_output, t_block_table owned by caches — do NOT destroy.
    // t_key, t_value owned by TensorLists.
    aclDestroyTensorList(key_list);
    aclDestroyTensorList(val_list);
    aclDestroyIntArray(seq_kv);
  }

 private:
  bool paged_ = false;

  mutable ascend::AclTensorCache prefill_q_cache_;

  mutable ascend::AclTensorCache prefill_out_cache_;

  mutable ascend::AclTensorCache decode_q_cache_;

  mutable ascend::AclTensorCache decode_out_cache_;

  mutable ascend::AclTensorCache block_table_cache_;

  aclTensor* causal_mask_ = nullptr;

  void* causal_mask_buf_ = nullptr;

  std::vector<int64_t> kv_shape_;

  std::vector<int64_t> kv_strides_;

  std::vector<int64_t> kv_storage_shape_;

  aclDataType kv_acl_dt_ = ACL_DT_UNDEFINED;
};

}  // namespace infini::ops

#endif
