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

// Allocate a 2048x2048 lower-triangular UINT8 causal mask on device.
// Required for sparseMode >= 2. Returns the aclTensor and sets `mask_buf`
// to the device allocation (caller must free both).
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
      std::optional<Tensor> block_table,
      std::optional<Tensor> cu_seqlens_q,
      std::optional<Tensor> cu_seqlens_kv,
      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
      double scale, int64_t sparse_mode, int64_t block_size,
      Tensor output) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    const int64_t N = query.size(1);
    const int64_t D = query.size(2);
    const int64_t KV_N = key.size(1);
    const size_t elem_bytes = query.element_size();
    const aclDataType acl_dtype = ascend::toAclDtype(query.dtype());

    bool paged = block_table.has_value() && block_size > 0;

    // --- Multi-sequence prefill ---
    // FIA cannot process token-packed multi-sequence TND directly with a single
    // BNSD call. Instead, call FIA once per sequence using pointer offsets into
    // the packed buffer.
    if (!paged && cu_seqlens_q.has_value()) {
      auto n_cu = cu_seqlens_q->numel();
      std::vector<int64_t> cu_q(n_cu), cu_kv(n_cu);
      aclrtMemcpyAsync(cu_q.data(), n_cu * sizeof(int64_t),
                       cu_seqlens_q->data(), n_cu * sizeof(int64_t),
                       ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (cu_seqlens_kv.has_value()) {
        aclrtMemcpyAsync(cu_kv.data(), n_cu * sizeof(int64_t),
                         cu_seqlens_kv->data(), n_cu * sizeof(int64_t),
                         ACL_MEMCPY_DEVICE_TO_HOST, stream);
      } else {
        cu_kv = cu_q;
      }
      aclrtSynchronizeStream(stream);

      // Build causal mask once and reuse across sequences.
      aclTensor* atten_mask = nullptr;
      void* mask_buf = nullptr;
      if (sparse_mode >= 2) {
        atten_mask = detail::makeCausalMask(&mask_buf, stream);
      }

      const int64_t num_seqs = n_cu - 1;
      for (int64_t s = 0; s < num_seqs; ++s) {
        const int64_t off_q = cu_q[s];
        const int64_t len_q = cu_q[s + 1] - off_q;
        const int64_t off_kv = cu_kv[s];
        const int64_t len_kv = cu_kv[s + 1] - off_kv;

        // BNSD input tensor for a TND slice starting at `tok_off`.
        auto make_bnsd = [&](const void* base, int64_t tok_off,
                             int64_t n_h, int64_t seq_len) -> aclTensor* {
          void* ptr = static_cast<uint8_t*>(const_cast<void*>(base))
                      + tok_off * n_h * D * static_cast<int64_t>(elem_bytes);
          std::vector<int64_t> sh = {1, n_h, seq_len, D};
          std::vector<int64_t> st = {n_h * seq_len * D, D, n_h * D, 1};
          std::vector<int64_t> ss = {n_h * seq_len * D};
          return aclCreateTensor(sh.data(), 4, acl_dtype, st.data(), 0,
                                 ACL_FORMAT_ND, ss.data(), 1, ptr);
        };
        // BSND output tensor: FIA writes BSND-contiguous = TND layout.
        auto make_bsnd = [&](void* base, int64_t tok_off,
                             int64_t n_h, int64_t seq_len) -> aclTensor* {
          void* ptr = static_cast<uint8_t*>(base)
                      + tok_off * n_h * D * static_cast<int64_t>(elem_bytes);
          std::vector<int64_t> sh = {1, seq_len, n_h, D};
          std::vector<int64_t> st = {seq_len * n_h * D, n_h * D, D, 1};
          std::vector<int64_t> ss = {seq_len * n_h * D};
          return aclCreateTensor(sh.data(), 4, acl_dtype, st.data(), 0,
                                 ACL_FORMAT_ND, ss.data(), 1, ptr);
        };

        aclTensor* t_q = make_bnsd(query.data(), off_q, N, len_q);
        aclTensor* t_k = make_bnsd(key.data(), off_kv, KV_N, len_kv);
        aclTensor* t_v = make_bnsd(value.data(), off_kv, KV_N, len_kv);
        aclTensor* t_out = make_bsnd(output.data(), off_q, N, len_q);

        const aclTensor* k_arr[] = {t_k};
        const aclTensor* v_arr[] = {t_v};
        aclTensorList* key_list = aclCreateTensorList(k_arr, 1);
        aclTensorList* val_list = aclCreateTensorList(v_arr, 1);

        int64_t ql = len_q, kvl = len_kv;
        aclIntArray* seq_q_ia = aclCreateIntArray(&ql, 1);
        aclIntArray* seq_kv_ia = aclCreateIntArray(&kvl, 1);

        uint64_t ws_needed = 0;
        aclOpExecutor* executor = nullptr;
        aclError gws = aclnnFusedInferAttentionScoreGetWorkspaceSize(
            t_q, key_list, val_list,
            nullptr, atten_mask,
            seq_q_ia, seq_kv_ia,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr,
            nullptr, nullptr, nullptr,
            num_heads, scale,
            static_cast<int64_t>(2147483647), static_cast<int64_t>(2147483647),
            const_cast<char*>("BNSD_BSND"),
            num_kv_heads, sparse_mode, 0, 0, 0, false,
            t_out, nullptr,
            &ws_needed, &executor);
        assert(gws == ACL_SUCCESS &&
               "aclnnFusedInferAttentionScoreGetWorkspaceSize failed (multi-seq)");

        auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
        aclError ret = aclnnFusedInferAttentionScore(
            arena.buf, ws_needed, executor, stream);
        assert(ret == ACL_SUCCESS &&
               "aclnnFusedInferAttentionScore failed (multi-seq)");

        aclDestroyTensor(t_q);
        aclDestroyTensor(t_out);
        aclDestroyTensorList(key_list);
        aclDestroyTensorList(val_list);
        aclDestroyIntArray(seq_q_ia);
        aclDestroyIntArray(seq_kv_ia);
      }

      if (atten_mask) aclDestroyTensor(atten_mask);
      if (mask_buf) aclrtFree(mask_buf);
      return;
    }

    // --- Single-sequence prefill or paged decode ---
    int64_t T = query.size(0);

    // Determine B based on whether this is paged decode or prefill.
    // Paged decode: B = num_reqs (block_table.size(0)), S_q = 1.
    // Prefill: B = 1, S_q = T (all tokens concatenated).
    int64_t B = paged ? block_table->size(0) : 1;
    int64_t S_q = paged ? 1 : T;

    // For prefill (S_q > 1): use inputLayout "BNSD_BSND".
    //   FIA reads inputs as BNSD [1, N, T, D] and writes output as BSND
    //   [1, T, N, D]. BSND-contiguous output is identical to TND memory layout,
    //   so no post-processing is needed.
    // For decode (S_q = 1, paged): use inputLayout "BNSD".
    //   With S_q=1, BNSD and BSND have the same memory layout, so output lands
    //   correctly without any transposition.
    const char* input_layout = paged ? "BNSD" : "BNSD_BSND";

    aclTensor* t_query = nullptr;
    aclTensor* t_output = nullptr;
    if (paged) {
      // Decode: query [B, N, D] viewed as BNSD [B, N, 1, D].
      std::vector<int64_t> q_shape = {B, N, S_q, D};
      std::vector<int64_t> q_strides = {N * D, D, D, 1};
      t_query = detail::reshapeView(query, q_shape, q_strides);
      t_output = detail::reshapeView(output, q_shape, q_strides);
    } else {
      // Prefill: query [T, N, D] viewed as BNSD [1, N, T, D].
      // Strides follow the TND memory layout: stride_N=D, stride_T=N*D.
      std::vector<int64_t> q_shape = {1, N, T, D};
      std::vector<int64_t> q_strides = {N * T * D, D, N * D, 1};
      t_query = detail::reshapeView(query, q_shape, q_strides);
      // Output is declared as BSND [1, T, N, D] (BSND-contiguous = TND layout).
      std::vector<int64_t> out_shape = {1, T, N, D};
      std::vector<int64_t> out_strides = {T * N * D, N * D, D, 1};
      t_output = detail::reshapeView(output, out_shape, out_strides);
    }

    // --- Handle key/value ---
    // For paged decode: key/value are the paged cache, used directly.
    // For prefill: key/value are TND [T, KV_N, D], viewed as BNSD [1, KV_N, T, D]
    // with TND strides (stride_N=D, stride_T=KV_N*D).
    aclTensor* t_key = nullptr;
    aclTensor* t_value = nullptr;
    if (paged) {
      t_key = ascend::buildAclTensor(key);
      t_value = ascend::buildAclTensor(value);
    } else {
      std::vector<int64_t> kv_shape = {1, KV_N, T, D};
      std::vector<int64_t> kv_strides = {KV_N * T * D, D, KV_N * D, 1};
      t_key = detail::reshapeView(key, kv_shape, kv_strides);
      t_value = detail::reshapeView(value, kv_shape, kv_strides);
    }

    // --- actualSeqLengths -> aclIntArray ---
    // Paged decode requires actualSeqLengthsKv (mandatory per CANN docs).
    // Single-sequence prefill supplies [T] for both Q and KV.
    aclIntArray* seq_q = nullptr;
    aclIntArray* seq_kv = nullptr;
    if (cu_seqlens_q.has_value()) {
      seq_q = detail::extractSeqLengths(cu_seqlens_q.value(), stream);
    } else if (!paged) {
      int64_t len_q = T;
      seq_q = aclCreateIntArray(&len_q, 1);
    }
    if (cu_seqlens_kv.has_value()) {
      seq_kv = detail::extractSeqLengths(cu_seqlens_kv.value(), stream);
    } else if (!paged) {
      int64_t len_kv = T;
      seq_kv = aclCreateIntArray(&len_kv, 1);
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

    // Nullptr for unused parameters (quantization, padding).
    aclTensor* pse_shift = nullptr;
    aclTensor* softmax_lse = nullptr;

    // sparseMode is invalid when Q_S=1 (paged decode); override to 0 to avoid
    // undefined FIA behaviour on this code path.
    const int64_t effective_sparse = paged ? 0 : sparse_mode;

    // sparseMode 2/3/4 require a 2048x2048 lower-triangular UINT8 template
    // mask; passing null triggers CANN error 561002.
    aclTensor* atten_mask = nullptr;
    void* mask_buf = nullptr;
    if (effective_sparse >= 2) {
      atten_mask = detail::makeCausalMask(&mask_buf, stream);
    }

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
    aclError gws_ret = aclnnFusedInferAttentionScoreGetWorkspaceSize(
        t_query,
        key_list,
        value_list,
        pse_shift,              // pseShift
        atten_mask,             // attenMask
        seq_q,                  // actualSeqLengths
        seq_kv,                 // actualSeqLengthsKv
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
        const_cast<char*>(input_layout),  // inputLayout
        num_kv_heads,           // numKeyValueHeads
        effective_sparse,       // sparseMode
        inner_precise,          // innerPrecise
        block_size,             // blockSize
        antiquant_mode,         // antiquantMode
        softmax_lse_flag,       // softmaxLseFlag (bool)
        t_output,               // attentionOut
        softmax_lse,            // softmaxLse
        &ws_needed,
        &executor);
    assert(gws_ret == ACL_SUCCESS &&
           "aclnnFusedInferAttentionScoreGetWorkspaceSize failed");

    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);

    aclError ret = aclnnFusedInferAttentionScore(arena.buf, ws_needed, executor, stream);
    assert(ret == ACL_SUCCESS && "aclnnFusedInferAttentionScore failed");

    // --- Cleanup ---
    // aclDestroyTensorList takes ownership of the contained tensor descriptors
    // (t_key, t_value), so destroy only the list — not the individual tensors.
    aclDestroyTensor(t_query);
    aclDestroyTensor(t_output);
    aclDestroyTensorList(key_list);
    aclDestroyTensorList(value_list);
    if (t_block_table) aclDestroyTensor(t_block_table);
    if (seq_q) aclDestroyIntArray(seq_q);
    if (seq_kv) aclDestroyIntArray(seq_kv);
    if (atten_mask) aclDestroyTensor(atten_mask);
    if (mask_buf) aclrtFree(mask_buf);
  }
};

}  // namespace infini::ops

#endif
