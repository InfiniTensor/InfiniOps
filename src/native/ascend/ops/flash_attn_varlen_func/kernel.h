#ifndef INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"
#include "base/flash_attn_varlen_func.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kAscend>
    : public FlashAttnVarlenFunc {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const int64_t max_seqlen_q, const int64_t max_seqlen_k, Tensor out)
      : Operator(q, k, v, cu_seqlens_q, cu_seqlens_k, std::nullopt,
                 std::nullopt, max_seqlen_q, max_seqlen_k, 0.0, std::nullopt,
                 false, {-1, -1}, 0.0, false, false, out, std::nullopt,
                 std::nullopt) {}

  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
           const int64_t max_seqlen_k, const double dropout_p,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool deterministic, const bool return_attn_probs, Tensor out,
           std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask)
      : FlashAttnVarlenFunc(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                            block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                            softmax_scale, causal, window_size, softcap,
                            deterministic, return_attn_probs, out, softmax_lse,
                            s_dmask),
        sequence_count_(cu_seqlens_q.numel() - 1),
        q_cache_(q),
        out_cache_(out) {
    ValidateSupportedOptions(alibi_slopes, return_attn_probs, softmax_lse,
                             s_dmask);
    ValidateTensors(q, k, v, block_table, out);

    cu_i32_host_.resize(sequence_count_ + 1);
    q_lengths_i64_.resize(sequence_count_);
    k_lengths_i64_.resize(sequence_count_);

    use_prompt_310p_ = ascend::IsAscend310P() && !block_table.has_value() &&
                       sequence_count_ == 1 && q.size(0) > 0 &&
                       q.dtype() == DataType::kFloat16 &&
                       q.size(2) == v.size(2) && q.size(0) == k.size(0) &&
                       q.size(0) == v.size(0) && causal &&
                       window_size == std::vector<int64_t>({-1, -1});
    if (use_prompt_310p_) {
      // The 310P prompt kernel's causal mask tiles require aligned sequences.
      constexpr int64_t kPromptSequenceAlignment = 256;
      padded_sequence_ = ((q.size(0) + kPromptSequenceAlignment - 1) /
                          kPromptSequenceAlignment) *
                         kPromptSequenceAlignment;
      prompt_q_cache_ = ascend::AclTensorCache(
          {1, padded_sequence_, static_cast<int64_t>(q.size(1) * q.size(2))},
          ACL_FLOAT16, nullptr);
      prompt_k_cache_ = ascend::AclTensorCache(
          {1, padded_sequence_, static_cast<int64_t>(k.size(1) * k.size(2))},
          ACL_FLOAT16, nullptr);
      prompt_v_cache_ = ascend::AclTensorCache(
          {1, padded_sequence_, static_cast<int64_t>(v.size(1) * v.size(2))},
          ACL_FLOAT16, nullptr);
      prompt_out_cache_ = ascend::AclTensorCache(
          {1, padded_sequence_,
           static_cast<int64_t>(out.size(1) * out.size(2))},
          ACL_FLOAT16, nullptr);
    }

    use_prompt_fallback_310p_ = ascend::IsAscend310P() && !use_prompt_310p_;
    if (!use_prompt_fallback_310p_ &&
        (causal || window_size != std::vector<int64_t>({-1, -1}))) {
      InitializeAttentionMask();
    }
  }

  ~Operator() override {
    if (!ascend::IsAclRuntimeAlive()) return;

    attention_mask_cache_.destroy();
    if (attention_mask_data_) aclrtFree(attention_mask_data_);
  }

  using FlashAttnVarlenFunc::operator();

  // CANN paged attention consumes a BnBsH cache. InfiniOps exposes the same
  // storage as BnBsND, so flatten the head dimensions in the ACL descriptor.
  static aclTensor* BuildPagedCacheAclTensor(const Tensor& tensor) {
    const std::vector<int64_t> shape{
        static_cast<int64_t>(tensor.size(0)),
        static_cast<int64_t>(tensor.size(1)),
        static_cast<int64_t>(tensor.size(2) * tensor.size(3)),
    };
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
         --index) {
      strides[index] = stride;
      stride *= shape[index];
    }
    const std::vector<int64_t> storage_shape{stride};
    return aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()),
                           ascend::ToAclDtype(tensor.dtype()), strides.data(),
                           /*storageOffset=*/0, ACL_FORMAT_ND,
                           storage_shape.data(),
                           static_cast<int64_t>(storage_shape.size()),
                           const_cast<void*>(tensor.data()));
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    ValidateSupportedOptions(alibi_slopes, return_attn_probs, softmax_lse,
                             s_dmask);
    ValidateTensors(q, k, v, block_table, out);
    const bool paged = block_table.has_value();

    auto stream = static_cast<aclrtStream>(stream_);
    auto actual_seq_lengths =
        MakeSequenceLengths(cu_seqlens_q, q_lengths_i64_, stream,
                            /*cumulative=*/true);
    auto actual_seq_lengths_kv =
        MakeSequenceLengths(cu_seqlens_k, k_lengths_i64_, stream,
                            /*cumulative=*/!block_table.has_value());
    if (use_prompt_fallback_310p_) {
      RunVariablePrompt310P(q, k, v, block_table, softmax_scale, causal,
                            window_size, out, stream);
      aclDestroyIntArray(actual_seq_lengths);
      aclDestroyIntArray(actual_seq_lengths_kv);
      return;
    }
    if (use_prompt_310p_) {
      assert(q_lengths_i64_[0] == static_cast<int64_t>(q.size(0)) &&
             k_lengths_i64_[0] == static_cast<int64_t>(k.size(0)) &&
             "310P prompt attention requires one complete sequence");
      RunPrompt310P(q, k, v, softmax_scale, out, stream);
      aclDestroyIntArray(actual_seq_lengths);
      aclDestroyIntArray(actual_seq_lengths_kv);
      return;
    }
    auto t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto t_k = paged ? BuildPagedCacheAclTensor(k) : ascend::BuildAclTensor(k);
    auto t_v = paged ? BuildPagedCacheAclTensor(v) : ascend::BuildAclTensor(v);
    auto t_out = out_cache_.get(out.data());
    auto t_attention_mask =
        attention_mask_data_ ? attention_mask_cache_.get(attention_mask_data_)
                             : nullptr;
    aclTensor* t_block_table = nullptr;
    if (block_table.has_value()) {
      t_block_table = ascend::BuildAclTensor(*block_table);
    }
    const aclTensor* keys[] = {t_k};
    const aclTensor* values[] = {t_v};
    auto key_list = aclCreateTensorList(keys, 1);
    auto value_list = aclCreateTensorList(values, 1);

    const auto max_token_count =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    auto pre_tokens = max_token_count;
    auto next_tokens = max_token_count;
    auto sparse_mode = int64_t{0};

    if (causal || window_size != std::vector<int64_t>({-1, -1})) {
      sparse_mode = window_size == std::vector<int64_t>({-1, -1}) ? 3 : 4;
      if (window_size[0] >= 0) pre_tokens = window_size[0];
      next_tokens = causal ? 0 : window_size[1];
      if (next_tokens < 0) next_tokens = max_token_count;
    }

    auto scale = softmax_scale.value_or(1.0 / std::sqrt(q.size(2)));
    const auto num_key_value_heads = paged ? k.size(2) : k.size(1);
    const auto block_size = paged ? k.size(1) : 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_q, key_list, value_list,
        /*pseShift=*/nullptr, t_attention_mask, actual_seq_lengths,
        actual_seq_lengths_kv,
        /*deqScale1=*/nullptr,
        /*quantScale1=*/nullptr,
        /*deqScale2=*/nullptr,
        /*quantScale2=*/nullptr,
        /*quantOffset2=*/nullptr,
        /*antiquantScale=*/nullptr,
        /*antiquantOffset=*/nullptr,
        /*blockTable=*/t_block_table,
        /*queryPaddingSize=*/nullptr,
        /*kvPaddingSize=*/nullptr,
        /*keyAntiquantScale=*/nullptr,
        /*keyAntiquantOffset=*/nullptr,
        /*valueAntiquantScale=*/nullptr,
        /*valueAntiquantOffset=*/nullptr,
        /*keySharedPrefix=*/nullptr,
        /*valueSharedPrefix=*/nullptr,
        /*actualSharedPrefixLen=*/nullptr,
        /*queryRope=*/nullptr,
        /*keyRope=*/nullptr,
        /*keyRopeAntiquantScale=*/nullptr,
        /*dequantScaleQuery=*/nullptr,
        /*learnableSink=*/nullptr, q.size(1), scale, pre_tokens, next_tokens,
        input_layout_.data(), num_key_value_heads, sparse_mode,
        // Correct fully masked rows when right-down causal has Q longer than
        // KV.
        /*innerPrecise=*/2,
        /*blockSize=*/block_size,
        /*antiquantMode=*/0,
        /*softmaxLseFlag=*/false,
        /*keyAntiquantMode=*/0,
        /*valueAntiquantMode=*/0,
        /*queryQuantMode=*/0, t_out,
        /*softmaxLse=*/nullptr, &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` workspace query failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size, "flash_attn_varlen_func_workspace");
    ret = aclnnFusedInferAttentionScoreV4(arena.buf, workspace_size, executor,
                                          stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` execution failed");

    aclDestroyTensorList(key_list);
    aclDestroyTensorList(value_list);
    if (t_block_table) aclDestroyTensor(t_block_table);
    aclDestroyIntArray(actual_seq_lengths);
    aclDestroyIntArray(actual_seq_lengths_kv);

    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)dropout_p;
    (void)softcap;
    (void)deterministic;
  }

 private:
  void RunVariablePrompt310P(const Tensor q, const Tensor k, const Tensor v,
                             const std::optional<Tensor>& block_table,
                             const std::optional<double> scale, bool causal,
                             const std::vector<int64_t>& window_size,
                             Tensor out, aclrtStream stream) const {
    assert(q.dtype() == DataType::kFloat16 &&
           "310P prompt attention requires FP16 tensors.");
    const bool paged = block_table.has_value();
    const int64_t kv_heads = paged ? k.size(2) : k.size(1);
    const size_t q_row_bytes = q.size(1) * q.size(2) * sizeof(uint16_t);
    const size_t k_row_bytes = kv_heads * q.size(2) * sizeof(uint16_t);
    const size_t v_row_bytes =
        kv_heads * v.size(v.ndim() - 1) * sizeof(uint16_t);
    std::vector<int32_t> blocks;
    if (paged) {
      blocks.resize(block_table->numel());
      ascend::CheckAcl(
          aclrtMemcpyAsync(blocks.data(), blocks.size() * sizeof(int32_t),
                           block_table->data(), blocks.size() * sizeof(int32_t),
                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
          "read 310P attention block table");
      ascend::CheckAcl(aclrtSynchronizeStream(stream),
                       "synchronize 310P attention block table");
    }
    int64_t q_begin = 0;
    int64_t k_begin = 0;
    for (size_t batch = 0; batch < sequence_count_; ++batch) {
      const int64_t q_end = q_lengths_i64_[batch];
      const int64_t q_length = q_end - q_begin;
      const int64_t k_length =
          paged ? k_lengths_i64_[batch] : k_lengths_i64_[batch] - k_begin;
      assert(q_length >= 0 && k_length >= 0 &&
             q_end <= static_cast<int64_t>(q.size(0)) &&
             "Invalid 310P attention sequence lengths.");
      auto* output = static_cast<char*>(out.data()) + q_begin * q_row_bytes;
      if (q_length == 0 || k_length == 0) {
        if (q_length > 0) {
          ascend::CheckAcl(aclrtMemsetAsync(output, q_length * q_row_bytes, 0,
                                            q_length * q_row_bytes, stream),
                           "zero 310P attention with empty keys");
        }
        q_begin = q_end;
        if (!paged) k_begin += k_length;
        continue;
      }
      // The 310P prompt kernel requires aligned sequence tiles. An explicit
      // mask preserves right-down causal/window semantics after padding.
      constexpr int64_t kAlignment = 256;
      const int64_t padded_length =
          ((std::max(q_length, k_length) + kAlignment - 1) / kAlignment) *
          kAlignment;
      const size_t q_bytes = padded_length * q_row_bytes;
      const size_t k_bytes = padded_length * k_row_bytes;
      const size_t v_bytes = padded_length * v_row_bytes;
      const size_t out_bytes = padded_length * q_row_bytes;
      const size_t total_bytes = q_bytes + k_bytes + v_bytes + out_bytes;
      auto& staging = ascend::GetWorkspacePool().Ensure(
          stream, total_bytes, "variable_prompt_310p_qkv_out");
      auto* q_data = static_cast<char*>(staging.buf);
      auto* k_data = q_data + q_bytes;
      auto* v_data = k_data + k_bytes;
      auto* out_data = v_data + v_bytes;
      ascend::CheckAcl(
          aclrtMemsetAsync(staging.buf, total_bytes, 0, total_bytes, stream),
          "zero 310P variable attention staging");
      const auto copy = [stream](void* dst, const void* src, size_t bytes) {
        ascend::CheckAcl(aclrtMemcpyAsync(dst, bytes, src, bytes,
                                          ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                         "stage 310P variable attention");
      };
      copy(q_data, static_cast<const char*>(q.data()) + q_begin * q_row_bytes,
           q_length * q_row_bytes);
      if (paged) {
        const int64_t page_size = k.size(1);
        const int64_t page_count = (k_length + page_size - 1) / page_size;
        assert(page_count <= static_cast<int64_t>(block_table->size(1)) &&
               "310P attention block table is too short.");
        for (int64_t page = 0; page < page_count; ++page) {
          const int64_t block = blocks[batch * block_table->size(1) + page];
          assert(block >= 0 && block < static_cast<int64_t>(k.size(0)) &&
                 "Invalid active 310P attention cache block.");
          const int64_t rows = std::min(page_size, k_length - page * page_size);
          copy(k_data + page * page_size * k_row_bytes,
               static_cast<const char*>(k.data()) +
                   block * page_size * k_row_bytes,
               rows * k_row_bytes);
          copy(v_data + page * page_size * v_row_bytes,
               static_cast<const char*>(v.data()) +
                   block * page_size * v_row_bytes,
               rows * v_row_bytes);
        }
      } else {
        assert(k_begin + k_length <= static_cast<int64_t>(k.size(0)) &&
               "310P attention key lengths exceed storage.");
        copy(k_data, static_cast<const char*>(k.data()) + k_begin * k_row_bytes,
             k_length * k_row_bytes);
        copy(v_data, static_cast<const char*>(v.data()) + k_begin * v_row_bytes,
             k_length * v_row_bytes);
      }
      std::vector<uint8_t> mask(padded_length * padded_length, 1);
      std::vector<bool> empty_rows(q_length, true);
      for (int64_t row = 0; row < q_length; ++row) {
        const int64_t center = row + k_length - q_length;
        for (int64_t column = 0; column < k_length; ++column) {
          const bool masked =
              (causal && column > center) ||
              (window_size[0] >= 0 && column < center - window_size[0]) ||
              (window_size[1] >= 0 && column > center + window_size[1]);
          mask[row * padded_length + column] = masked;
          if (!masked) empty_rows[row] = false;
        }
      }
      // Avoid all-masked softmax rows inside CANN; discard these results below.
      for (int64_t row = 0; row < padded_length; ++row) {
        if (row >= q_length || empty_rows[row]) mask[row * padded_length] = 0;
      }
      auto& mask_buffer = ascend::GetWorkspacePool().Ensure(
          stream, mask.size(), "variable_prompt_310p_mask");
      ascend::CheckAcl(
          aclrtMemcpyAsync(mask_buffer.buf, mask.size(), mask.data(),
                           mask.size(), ACL_MEMCPY_HOST_TO_DEVICE, stream),
          "upload 310P variable attention mask");
      ascend::AclTensorCache t_q(
          {1, padded_length, static_cast<int64_t>(q.size(1) * q.size(2))},
          ACL_FLOAT16, q_data);
      ascend::AclTensorCache t_k(
          {1, padded_length, kv_heads * static_cast<int64_t>(q.size(2))},
          ACL_FLOAT16, k_data);
      ascend::AclTensorCache t_v(
          {1, padded_length,
           kv_heads * static_cast<int64_t>(v.size(v.ndim() - 1))},
          ACL_FLOAT16, v_data);
      ascend::AclTensorCache t_out(
          {1, padded_length, static_cast<int64_t>(q.size(1) * q.size(2))},
          ACL_FLOAT16, out_data);
      ascend::AclTensorCache t_mask({padded_length, padded_length}, ACL_BOOL,
                                    mask_buffer.buf);
      uint64_t workspace_bytes = 0;
      aclOpExecutor* executor = nullptr;
      char layout[] = "BSH";
      ascend::CheckAcl(
          aclnnPromptFlashAttentionV3GetWorkspaceSize(
              t_q.get(q_data), t_k.get(k_data), t_v.get(v_data), nullptr,
              t_mask.get(mask_buffer.buf), nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, q.size(1),
              scale.value_or(1.0 / std::sqrt(q.size(2))),
              std::numeric_limits<int32_t>::max(),
              std::numeric_limits<int32_t>::max(), layout, kv_heads, 0, 0,
              t_out.get(out_data), &workspace_bytes, &executor),
          "query 310P variable PromptFlashAttentionV3 workspace");
      auto& workspace = ascend::GetWorkspacePool().Ensure(
          stream, workspace_bytes, "variable_prompt_310p_workspace");
      ascend::CheckAcl(aclnnPromptFlashAttentionV3(
                           workspace.buf, workspace_bytes, executor, stream),
                       "run 310P variable PromptFlashAttentionV3");
      copy(output, out_data, q_length * q_row_bytes);
      for (int64_t row = 0; row < q_length; ++row) {
        if (empty_rows[row]) {
          ascend::CheckAcl(
              aclrtMemsetAsync(output + row * q_row_bytes, q_row_bytes, 0,
                               q_row_bytes, stream),
              "zero fully masked 310P attention row");
        }
      }
      // Keep the host mask and ACL descriptors alive until this sequence ends.
      ascend::CheckAcl(aclrtSynchronizeStream(stream),
                       "synchronize 310P variable prompt attention");
      q_begin = q_end;
      if (!paged) k_begin += k_length;
    }
  }

  static void ValidateSupportedOptions(
      const std::optional<Tensor>& alibi_slopes, bool return_attn_probs,
      const std::optional<Tensor>& softmax_lse,
      const std::optional<Tensor>& s_dmask) {
    assert(!alibi_slopes.has_value() &&
           "Ascend FlashAttnVarlenFunc does not support ALiBi");
    assert(!return_attn_probs && !softmax_lse.has_value() &&
           !s_dmask.has_value() &&
           "Ascend `FlashAttnVarlenFunc` does not support auxiliary outputs");
  }

  static void ValidateTensors(const Tensor q, const Tensor k, const Tensor v,
                              const std::optional<Tensor>& block_table,
                              const Tensor out) {
    const bool paged = block_table.has_value();
    assert(q.IsContiguous() && k.IsContiguous() && v.IsContiguous() &&
           out.IsContiguous() &&
           "Ascend FlashAttnVarlenFunc requires contiguous tensors");
    assert(q.ndim() == 3 && out.ndim() == 3 &&
           ((paged && k.ndim() == 4 && v.ndim() == 4) ||
            (!paged && k.ndim() == 3 && v.ndim() == 3)) &&
           "Ascend FlashAttnVarlenFunc expects dense TND key/value or "
           "paged block key/value tensors");
    if (paged) {
      assert(block_table->ndim() == 2 && block_table->IsContiguous() &&
             block_table->dtype() == DataType::kInt32 && k.size(1) > 0 &&
             "Ascend paged attention requires a contiguous int32 block "
             "table and a positive cache block size");
    }
  }

  void InitializeAttentionMask() {
    const int64_t mask_size = use_prompt_310p_ ? padded_sequence_ : 2048;
    std::vector<uint8_t> mask(mask_size * mask_size);

    for (int64_t row = 0; row < mask_size; ++row) {
      for (int64_t column = row + 1; column < mask_size; ++column) {
        mask[row * mask_size + column] = 1;
      }
    }

    const auto bytes = mask.size() * sizeof(mask[0]);
    auto ret =
        aclrtMalloc(&attention_mask_data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to allocate attention mask");
    ret = aclrtMemcpy(attention_mask_data_, bytes, mask.data(), bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to upload attention mask");
    attention_mask_cache_ = ascend::AclTensorCache(
        {mask_size, mask_size}, ACL_BOOL, attention_mask_data_);
  }

  void RunPrompt310P(const Tensor q, const Tensor k, const Tensor v,
                     const std::optional<double> scale, Tensor out,
                     aclrtStream stream) const {
    const size_t q_row_bytes = q.size(1) * q.size(2) * sizeof(uint16_t);
    const size_t k_row_bytes = k.size(1) * k.size(2) * sizeof(uint16_t);
    const size_t v_row_bytes = v.size(1) * v.size(2) * sizeof(uint16_t);
    const size_t q_bytes = padded_sequence_ * q_row_bytes;
    const size_t k_bytes = padded_sequence_ * k_row_bytes;
    const size_t v_bytes = padded_sequence_ * v_row_bytes;
    const size_t total_bytes = q_bytes * 2 + k_bytes + v_bytes;
    auto& padded = ascend::GetWorkspacePool().Ensure(
        stream, total_bytes, "prompt_310p_padded_qkv_out");
    auto* q_data = static_cast<char*>(padded.buf);
    auto* k_data = q_data + q_bytes;
    auto* v_data = k_data + k_bytes;
    auto* out_data = v_data + v_bytes;
    ascend::CheckAcl(
        aclrtMemsetAsync(padded.buf, total_bytes, 0, total_bytes, stream),
        "zero padded prompt QKV");
    ascend::CheckAcl(
        aclrtMemcpyAsync(q_data, q_bytes, q.data(), q.size(0) * q_row_bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
        "pad prompt Q");
    ascend::CheckAcl(
        aclrtMemcpyAsync(k_data, k_bytes, k.data(), k.size(0) * k_row_bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
        "pad prompt K");
    ascend::CheckAcl(
        aclrtMemcpyAsync(v_data, v_bytes, v.data(), v.size(0) * v_row_bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
        "pad prompt V");
    char layout[] = "BSH";
    uint64_t workspace_bytes = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnPromptFlashAttentionV3GetWorkspaceSize(
        prompt_q_cache_.get(q_data), prompt_k_cache_.get(k_data),
        prompt_v_cache_.get(v_data), nullptr,
        attention_mask_cache_.get(attention_mask_data_), nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, q.size(1),
        scale.value_or(1.0 / std::sqrt(q.size(2))),
        std::numeric_limits<int32_t>::max(), 0, layout, k.size(1), 0, 0,
        prompt_out_cache_.get(out_data), &workspace_bytes, &executor);
    ascend::CheckAcl(status, "query 310P PromptFlashAttentionV3 workspace");
    auto& workspace = ascend::GetWorkspacePool().Ensure(
        stream, workspace_bytes, "prompt_310p_attention_workspace");
    ascend::CheckAcl(aclnnPromptFlashAttentionV3(workspace.buf, workspace_bytes,
                                                 executor, stream),
                     "run 310P PromptFlashAttentionV3");
    const auto output_bytes = q.size(0) * q_row_bytes;
    ascend::CheckAcl(
        aclrtMemcpyAsync(out.data(), output_bytes, out_data, output_bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
        "unpad prompt output");
  }

  // Dense TND K/V uses cumulative endpoints, while paged K/V uses per-batch
  // lengths. Query is always dense TND and therefore cumulative.
  aclIntArray* MakeSequenceLengths(const Tensor cu_seqlens,
                                   std::vector<int64_t>& lengths,
                                   aclrtStream stream, bool cumulative) const {
    const auto bytes = cu_i32_host_.size() * sizeof(cu_i32_host_[0]);
    auto ret = aclrtMemcpyAsync(cu_i32_host_.data(), bytes, cu_seqlens.data(),
                                bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to copy sequence lengths");
    ret = aclrtSynchronizeStream(stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to synchronize lengths");

    std::transform(
        cu_i32_host_.begin() + 1, cu_i32_host_.end(), cu_i32_host_.begin(),
        lengths.begin(), [cumulative](int32_t current, int32_t previous) {
          return static_cast<int64_t>(cumulative ? current
                                                 : current - previous);
        });
    return aclCreateIntArray(lengths.data(), lengths.size());
  }

  size_t sequence_count_{0};
  mutable ascend::AclTensorCache q_cache_;
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache attention_mask_cache_;
  mutable std::vector<int32_t> cu_i32_host_;
  mutable std::vector<int64_t> q_lengths_i64_;
  mutable std::vector<int64_t> k_lengths_i64_;
  void* attention_mask_data_{nullptr};
  bool use_prompt_310p_{false};
  bool use_prompt_fallback_310p_{false};
  int64_t padded_sequence_{0};
  mutable ascend::AclTensorCache prompt_q_cache_;
  mutable ascend::AclTensorCache prompt_k_cache_;
  mutable ascend::AclTensorCache prompt_v_cache_;
  mutable ascend::AclTensorCache prompt_out_cache_;
  mutable std::array<char, 4> input_layout_{'T', 'N', 'D', '\0'};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
