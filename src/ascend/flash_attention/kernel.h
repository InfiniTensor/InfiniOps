#ifndef INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "ascend/device.h"
#include "base/flash_attention.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Build an aclTensor with a different view shape/stride but the same data
// pointer.
inline aclTensor* reshapeView(const Tensor& t,
                              const std::vector<int64_t>& new_shape,
                              const std::vector<int64_t>& new_strides) {
    int64_t storage_elems = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == 0) { storage_elems = 0; break; }
        if (new_strides[i] > 0 && new_shape[i] > 1) {
            storage_elems += static_cast<int64_t>(new_shape[i] - 1) * new_strides[i];
        }
    }
    std::vector<int64_t> storage_shape = {storage_elems};
    return aclCreateTensor(
        new_shape.data(), static_cast<int64_t>(new_shape.size()),
        ascend::toAclDtype(t.dtype()),
        new_strides.data(), 0, ACL_FORMAT_ND,
        storage_shape.data(), static_cast<int64_t>(storage_shape.size()),
        const_cast<void*>(t.data()));
}

// Extract cu_seqlens differences to a host aclIntArray.
// cu_seqlens = [0, s1, s1+s2, ...] -> per_seq_lens = [s1, s2, ...].
// Used by paged decode (actualSeqLengthsKv = per-sequence KV lengths).
inline aclIntArray* extractSeqLengths(const Tensor& cu_seqlens, aclrtStream stream) {
    auto n = cu_seqlens.numel();
    std::vector<int64_t> cu_host(n);
    aclrtMemcpyAsync(cu_host.data(), n * sizeof(int64_t),
                     cu_seqlens.data(), n * sizeof(int64_t),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    std::vector<int64_t> lengths(n - 1);
    for (size_t i = 0; i < lengths.size(); ++i) {
        lengths[i] = cu_host[i + 1] - cu_host[i];
    }
    return aclCreateIntArray(lengths.data(), static_cast<int64_t>(lengths.size()));
}

// Extract cumulative end positions from cu_seqlens to a host aclIntArray.
// cu_seqlens = [0, s1, s1+s2, ...] -> cum_lens = [s1, s1+s2, ...].
// FIA V4 TND varlen uses cumulative end positions, matching the vllm-ascend
// convention for npu_fused_infer_attention_score actual_seq_lengths.
inline aclIntArray* cumSeqLengths(const Tensor& cu_seqlens, aclrtStream stream) {
    auto n = cu_seqlens.numel();
    std::vector<int64_t> cu_host(n);
    aclrtMemcpyAsync(cu_host.data(), n * sizeof(int64_t),
                     cu_seqlens.data(), n * sizeof(int64_t),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    // Skip the leading 0; return [s1, s1+s2, ...].
    return aclCreateIntArray(cu_host.data() + 1,
                             static_cast<int64_t>(n - 1));
}

// Allocate a 2048x2048 lower-triangular UINT8 causal mask on device.
// Required for sparseMode >= 2.
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
    return aclCreateTensor(mask_shape.data(), 2, ACL_UINT8,
                           mask_strides.data(), 0, ACL_FORMAT_ND,
                           mask_storage.data(), 1, *mask_buf);
}

}  // namespace detail

template <>
class Operator<FlashAttention, Device::Type::kAscend> : public FlashAttention {
 public:
  using FlashAttention::FlashAttention;

  void operator()(
      const Tensor query, const Tensor key, const Tensor value,
      std::optional<Tensor> cu_seqlens_q,
      std::optional<Tensor> cu_seqlens_kv,
      std::optional<Tensor> block_table,
      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
      double scale, bool causal, int64_t window_left, int64_t window_right,
      int64_t block_size, Tensor output) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    const bool paged = block_table.has_value() && block_size > 0;

    // Map causal + window_left/right to FIA sparse_mode / preTokens / nextTokens.
    //
    //   causal=true, window_left<0              -> sparse_mode=3 (full causal)
    //   causal=true, window_left>=0             -> sparse_mode=4 (sliding window causal)
    //   causal=false                            -> sparse_mode=0 (no mask)
    //
    // sparse_mode is ignored by FIA when Q_S=1 (paged decode); effective_sparse
    // is set to 0 in that path to avoid allocating the unnecessary causal mask.
    int64_t sparse_mode;
    int64_t pre_tokens  = 2147483647;
    int64_t next_tokens = 2147483647;
    if (causal) {
      if (window_left >= 0) {
        sparse_mode = 4;  // band: sliding window causal
        pre_tokens  = window_left;
        next_tokens = 0;
      } else {
        sparse_mode = 3;  // rightDownCausal: full causal, pre/next ignored
        next_tokens = 0;
      }
    } else {
      sparse_mode = 0;
      if (window_left  >= 0) pre_tokens  = window_left;
      if (window_right >= 0) next_tokens = window_right;
    }

    if (!paged) {
      // --- Prefill (single- or multi-sequence) ---
      // V4 TND: query/key/value passed as token-packed [T, N, D]; per-sequence
      // lengths are derived from cu_seqlens. Single fused call for all sequences,
      // equivalent to flash_attn_varlen_func on CUDA.
      int64_t T = query.size(0);

      // V4 TND varlen uses cumulative end positions [s1, s1+s2, ...].
      // For single-seq (no cu_seqlens), [T] is both per-seq and cumulative.
      aclIntArray* seq_q = cu_seqlens_q.has_value()
          ? detail::cumSeqLengths(cu_seqlens_q.value(), stream)
          : aclCreateIntArray(&T, 1);
      aclIntArray* seq_kv = cu_seqlens_kv.has_value()
          ? detail::cumSeqLengths(cu_seqlens_kv.value(), stream)
          : aclCreateIntArray(&T, 1);

      aclTensor* t_q   = ascend::buildAclTensor(query);
      aclTensor* t_k   = ascend::buildAclTensor(key);
      aclTensor* t_v   = ascend::buildAclTensor(value);
      aclTensor* t_out = ascend::buildAclTensor(output);

      const aclTensor* k_arr[] = {t_k};
      const aclTensor* v_arr[] = {t_v};
      aclTensorList* key_list = aclCreateTensorList(k_arr, 1);
      aclTensorList* val_list = aclCreateTensorList(v_arr, 1);

      // sparseMode 2/3/4 require a 2048x2048 lower-triangular causal mask.
      aclTensor* atten_mask = nullptr;
      void* mask_buf = nullptr;
      if (sparse_mode >= 2) {
        atten_mask = detail::makeCausalMask(&mask_buf, stream);
      }

      uint64_t ws_needed = 0;
      aclOpExecutor* executor = nullptr;
      // Parameter order: query, key, value,
      //   pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
      //   deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
      //   antiquantScale, antiquantOffset,
      //   blockTable, queryPaddingSize, kvPaddingSize,
      //   keyAntiquantScale, keyAntiquantOffset,
      //   valueAntiquantScale, valueAntiquantOffset,
      //   keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
      //   queryRope, keyRope, keyRopeAntiquantScale,
      //   dequantScaleQuery, learnableSink,
      //   numHeads, scaleValue, preTokens, nextTokens, inputLayout,
      //   numKeyValueHeads, sparseMode, innerPrecise, blockSize,
      //   antiquantMode, softmaxLseFlag,
      //   keyAntiquantMode, valueAntiquantMode, queryQuantMode,
      //   attentionOut, softmaxLse, workspaceSize, executor
      aclError gws = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
          t_q, key_list, val_list,
          nullptr,        // pseShift
          atten_mask,     // attenMask
          seq_q,          // actualSeqLengths
          seq_kv,         // actualSeqLengthsKv
          nullptr, nullptr, nullptr, nullptr, nullptr,  // deqScale1..quantOffset2
          nullptr, nullptr,                             // antiquantScale, antiquantOffset
          nullptr,                                      // blockTable
          nullptr, nullptr,                             // queryPaddingSize, kvPaddingSize
          nullptr, nullptr, nullptr, nullptr,           // key/value antiquant scale/offset
          nullptr, nullptr, nullptr,                    // keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen
          nullptr, nullptr, nullptr,                    // queryRope, keyRope, keyRopeAntiquantScale
          nullptr, nullptr,                             // dequantScaleQuery, learnableSink
          num_heads, scale,
          pre_tokens, next_tokens,
          const_cast<char*>("TND"),
          num_kv_heads, sparse_mode,
          0,              // innerPrecise
          0,              // blockSize (unused for prefill)
          0, false,       // antiquantMode, softmaxLseFlag
          0, 0, 0,        // keyAntiquantMode, valueAntiquantMode, queryQuantMode
          t_out, nullptr,
          &ws_needed, &executor);
      assert(gws == ACL_SUCCESS &&
             "aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed (prefill)");

      auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
      aclError ret = aclnnFusedInferAttentionScoreV4(
          arena.buf, ws_needed, executor, stream);
      assert(ret == ACL_SUCCESS &&
             "aclnnFusedInferAttentionScoreV4 failed (prefill)");

      aclDestroyTensor(t_q);
      aclDestroyTensor(t_out);
      aclDestroyTensorList(key_list);
      aclDestroyTensorList(val_list);
      aclDestroyIntArray(seq_q);
      aclDestroyIntArray(seq_kv);
      if (atten_mask) aclDestroyTensor(atten_mask);
      if (mask_buf) aclrtFree(mask_buf);
      return;
    }

    // --- Paged decode ---
    // V4 BNSD: reshape query/output [B, N, D] -> [B, N, 1, D].
    // KV cache [num_blocks, block_size, N_kv, D] flattened to
    // [num_blocks, block_size, N_kv*D] (zero-copy, FIA BSH kv format).
    assert(cu_seqlens_kv.has_value() &&
           "`FlashAttention` paged decode requires `cu_seqlens_kv`");

    const int64_t N   = query.size(1);
    const int64_t D   = query.size(2);
    const int64_t B   = query.size(0);
    const int64_t nb  = key.size(0);
    const int64_t bsz = key.size(1);
    const int64_t NkvD = key.size(2) * key.size(3);

    std::vector<int64_t> bnsd_sh = {B, N, 1, D};
    std::vector<int64_t> bnsd_st = {N * D, D, D, 1};
    aclTensor* t_query  = detail::reshapeView(query,  bnsd_sh, bnsd_st);
    aclTensor* t_output = detail::reshapeView(output, bnsd_sh, bnsd_st);

    std::vector<int64_t> kv_sh = {nb, bsz, NkvD};
    std::vector<int64_t> kv_st = {bsz * NkvD, NkvD, 1};
    aclTensor* t_key   = detail::reshapeView(key,   kv_sh, kv_st);
    aclTensor* t_value = detail::reshapeView(value, kv_sh, kv_st);

    aclIntArray* seq_kv = detail::extractSeqLengths(cu_seqlens_kv.value(), stream);
    aclTensor* t_block_table = ascend::buildAclTensor(block_table.value());

    const aclTensor* k_arr[] = {t_key};
    const aclTensor* v_arr[] = {t_value};
    aclTensorList* key_list = aclCreateTensorList(k_arr, 1);
    aclTensorList* val_list = aclCreateTensorList(v_arr, 1);

    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    aclError gws = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_query, key_list, val_list,
        nullptr,         // pseShift
        nullptr,         // attenMask (sparseMode ignored for Q_S=1)
        nullptr,         // actualSeqLengths (ignored for Q_S=1)
        seq_kv,          // actualSeqLengthsKv (mandatory for paged)
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr,
        t_block_table,   // blockTable
        nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr,
        num_heads, scale,
        static_cast<int64_t>(2147483647), static_cast<int64_t>(2147483647),
        const_cast<char*>("BNSD"),
        num_kv_heads,
        0,               // sparseMode=0 (ignored for Q_S=1)
        0,               // innerPrecise
        block_size,      // blockSize
        0, false,        // antiquantMode, softmaxLseFlag
        0, 0, 0,         // keyAntiquantMode, valueAntiquantMode, queryQuantMode
        t_output, nullptr,
        &ws_needed, &executor);
    assert(gws == ACL_SUCCESS &&
           "aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed (decode)");

    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclError ret = aclnnFusedInferAttentionScoreV4(
        arena.buf, ws_needed, executor, stream);
    assert(ret == ACL_SUCCESS &&
           "aclnnFusedInferAttentionScoreV4 failed (decode)");

    aclDestroyTensor(t_query);
    aclDestroyTensor(t_output);
    aclDestroyTensorList(key_list);
    aclDestroyTensorList(val_list);
    aclDestroyTensor(t_block_table);
    aclDestroyIntArray(seq_kv);
  }
};

}  // namespace infini::ops

#endif
