#ifndef INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTENTION_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score.h"
#include "ascend/device.h"
#include "base/flash_attention.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Build an aclTensor with a different view shape/stride but the same data
// pointer. Used to reshape TND [T,N,D] to BNSD [1,N,T,D] without copying.
inline aclTensor* reshapeView(const Tensor& t,
                              const std::vector<int64_t>& new_shape,
                              const std::vector<int64_t>& new_strides) {
    // Compute storage shape from the new strides.
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

// Extract cu_seqlens differences to an aclIntArray on the host.
// cu_seqlens = [0, s1, s1+s2, ...] -> actual_seq_lens = [s1, s2, ...].
inline aclIntArray* extractSeqLengths(const Tensor& cu_seqlens, aclrtStream stream) {
    auto n = cu_seqlens.numel();
    std::vector<int64_t> cu_host(n);
    aclrtMemcpyAsync(cu_host.data(), n * sizeof(int64_t),
                     cu_seqlens.data(), n * sizeof(int64_t),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    // Differences give per-request sequence lengths.
    std::vector<int64_t> lengths(n - 1);
    for (size_t i = 0; i < lengths.size(); ++i) {
        lengths[i] = cu_host[i + 1] - cu_host[i];
    }
    return aclCreateIntArray(lengths.data(), static_cast<int64_t>(lengths.size()));
}

}  // namespace detail

template <>
class Operator<FlashAttention, Device::Type::kAscend> : public FlashAttention {
 public:
  using FlashAttention::FlashAttention;

  void operator()(
      const Tensor query, const Tensor key, const Tensor value,
      std::optional<Tensor> block_table,
      std::optional<Tensor> cu_seqlens_q,
      std::optional<Tensor> cu_seqlens_kv,
      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
      double scale, int64_t sparse_mode, int64_t block_size,
      Tensor output) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    int64_t T = query.size(0);
    int64_t N = query.size(1);
    int64_t D = query.size(2);
    int64_t KV_N = key.size(1);

    // Determine B based on whether this is paged decode or prefill.
    // Paged decode: B = num_reqs (block_table.size(0)), S_q = 1.
    // Prefill: B = 1, S_q = T (all tokens concatenated).
    bool paged = block_table.has_value() && block_size > 0;
    int64_t B = paged ? block_table->size(0) : 1;
    int64_t S_q = paged ? 1 : T;

    // --- Reshape query TND -> BNSD ---
    // query [T, N, D] -> [B, N, S_q, D]
    std::vector<int64_t> q_bnsd_shape = {B, N, S_q, D};
    std::vector<int64_t> q_bnsd_strides = {N * S_q * D, S_q * D, D, 1};
    auto* t_query = detail::reshapeView(query, q_bnsd_shape, q_bnsd_strides);

    // output [T, N, D] -> [B, N, S_q, D]
    auto* t_output = detail::reshapeView(output, q_bnsd_shape, q_bnsd_strides);

    // --- Handle key/value ---
    // For paged decode: key/value are the paged cache, used directly.
    // For prefill: key/value are TND, reshape to [1, KV_N, T, D].
    aclTensor* t_key = nullptr;
    aclTensor* t_value = nullptr;
    if (paged) {
      t_key = ascend::buildAclTensor(key);
      t_value = ascend::buildAclTensor(value);
    } else {
      std::vector<int64_t> kv_bnsd_shape = {1, KV_N, T, D};
      std::vector<int64_t> kv_bnsd_strides = {KV_N * T * D, T * D, D, 1};
      t_key = detail::reshapeView(key, kv_bnsd_shape, kv_bnsd_strides);
      t_value = detail::reshapeView(value, kv_bnsd_shape, kv_bnsd_strides);
    }

    // --- cu_seqlens -> aclIntArray ---
    aclIntArray* seq_q = nullptr;
    aclIntArray* seq_kv = nullptr;
    if (cu_seqlens_q.has_value()) {
      seq_q = detail::extractSeqLengths(cu_seqlens_q.value(), stream);
    }
    if (cu_seqlens_kv.has_value()) {
      seq_kv = detail::extractSeqLengths(cu_seqlens_kv.value(), stream);
    }

    // --- block_table ---
    aclTensor* t_block_table = nullptr;
    if (block_table.has_value()) {
      t_block_table = ascend::buildAclTensor(block_table.value());
    }

    // --- Key/Value tensor list ---
    // FIA requires aclTensorList* for key/value. Single-element list.
    const aclTensor* k_tensors[] = {t_key};
    const aclTensor* v_tensors[] = {t_value};
    aclTensorList* key_list = aclCreateTensorList(k_tensors, 1);
    aclTensorList* value_list = aclCreateTensorList(v_tensors, 1);

    // --- Fixed defaults ---
    int64_t pre_tokens = 2147483647;   // INT_MAX
    int64_t next_tokens = 2147483647;  // INT_MAX
    int64_t inner_precise = 0;
    int64_t antiquant_mode = 0;
    bool softmax_lse_flag = false;

    // Nullptr for unused parameters (quantization, padding, masks).
    aclTensor* pse_shift = nullptr;
    aclTensor* atten_mask = nullptr;
    aclIntArray* actual_seq_q = seq_q;
    aclIntArray* actual_seq_kv = seq_kv;
    aclTensor* softmax_lse = nullptr;

    // --- Call FIA two-phase ---
    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;

    // Parameter order follows the CANN aclnnFusedInferAttentionScore API:
    //   query, key, value, pseShift, attenMask,
    //   actualSeqLengths, actualSeqLengthsKv,
    //   deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
    //   antiquantScale, antiquantOffset,
    //   blockTable, queryPaddingSize, kvPaddingSize,
    //   numHeads, scaleValue, preTokens, nextTokens, inputLayout,
    //   numKeyValueHeads, sparseMode, innerPrecise, blockSize,
    //   antiquantMode, softmaxLseFlag,
    //   attentionOut, softmaxLse,
    //   workspaceSize, executor
    aclnnFusedInferAttentionScoreGetWorkspaceSize(
        t_query,
        key_list,
        value_list,
        pse_shift,              // pseShift
        atten_mask,             // attenMask
        actual_seq_q,           // actualSeqLengths
        actual_seq_kv,          // actualSeqLengthsKv
        nullptr,                // deqScale1
        nullptr,                // quantScale1
        nullptr,                // deqScale2
        nullptr,                // quantScale2
        nullptr,                // quantOffset2
        nullptr,                // antiquantScale
        nullptr,                // antiquantOffset
        t_block_table,          // blockTable
        nullptr,                // queryPaddingSize
        nullptr,                // kvPaddingSize
        num_heads,              // numHeads
        scale,                  // scaleValue (double)
        pre_tokens,             // preTokens
        next_tokens,            // nextTokens
        const_cast<char*>("BNSD"),  // inputLayout
        num_kv_heads,           // numKeyValueHeads
        sparse_mode,            // sparseMode
        inner_precise,          // innerPrecise
        block_size,             // blockSize
        antiquant_mode,         // antiquantMode
        softmax_lse_flag,       // softmaxLseFlag (bool)
        t_output,               // attentionOut
        softmax_lse,            // softmaxLse
        &ws_needed,
        &executor);

    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);

    aclnnFusedInferAttentionScore(arena.buf, ws_needed, executor, stream);

    // --- Cleanup ---
    aclDestroyTensor(t_query);
    aclDestroyTensor(t_key);
    aclDestroyTensor(t_value);
    aclDestroyTensor(t_output);
    aclDestroyTensorList(key_list);
    aclDestroyTensorList(value_list);
    if (t_block_table) aclDestroyTensor(t_block_table);
    if (seq_q) aclDestroyIntArray(seq_q);
    if (seq_kv) aclDestroyIntArray(seq_kv);
  }
};

}  // namespace infini::ops

#endif
