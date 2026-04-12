#ifndef INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "base/flash_attention.h"
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/pos_enc.cuh"

namespace infini::ops {

// FlashAttention via FlashInfer header-only API.
//
// Supports four modes, selected by the presence of optional tensors:
// 1. Paged decode: `block_table` present — batch decode with paged KV cache
// 2. Batch prefill: `cu_seqlens_q` present — multiple packed sequences
// 3. Single decode: `num_tokens == 1` — single token, contiguous KV
// 4. Single prefill: default — single sequence, contiguous KV
//
// Batch prefill and paged decode use a per-sequence loop over the single-
// sequence kernels. This is functionally correct; a future optimization can
// switch to FlashInfer's native batch kernels with scheduler workspace.
template <typename Backend>
class CudaFlashAttention : public FlashAttention {
 public:
  using FlashAttention::FlashAttention;

  void operator()(const Tensor query, const Tensor key, const Tensor value,
                  std::optional<Tensor> cu_seqlens_q,
                  std::optional<Tensor> cu_seqlens_kv,
                  std::optional<Tensor> block_table, int64_t num_heads,
                  int64_t num_kv_heads, int64_t head_size, double scale,
                  bool causal, int64_t window_left, int64_t window_right,
                  int64_t block_size, Tensor output) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    if (block_table.has_value()) {
      // Paged decode: block_table present.
      DispatchHeadDimPagedDecode(query, key, value, cu_seqlens_q.value(),
                                 cu_seqlens_kv.value(), block_table.value(),
                                 output, num_heads, num_kv_heads, head_size,
                                 scale, window_left, block_size, cuda_stream);
    } else if (cu_seqlens_q.has_value()) {
      // Batch prefill: cu_seqlens present, packed sequences.
      auto mask_mode = causal ? flashinfer::MaskMode::kCausal
                              : flashinfer::MaskMode::kNone;
      DispatchHeadDimBatchPrefill(query, key, value, cu_seqlens_q.value(),
                                  cu_seqlens_kv.value(), output, num_heads,
                                  num_kv_heads, head_size, scale, window_left,
                                  mask_mode, cuda_stream);
    } else if (num_tokens_ == 1) {
      // Single decode: single token query, full KV cache.
      DispatchHeadDimDecode(query, key, value, output, num_heads, num_kv_heads,
                            head_size, scale, window_left, cuda_stream);
    } else if (causal) {
      DispatchHeadDimPrefill(query, key, value, output, num_heads, num_kv_heads,
                             head_size, scale, window_left,
                             flashinfer::MaskMode::kCausal, cuda_stream);
    } else {
      DispatchHeadDimPrefill(query, key, value, output, num_heads, num_kv_heads,
                             head_size, scale, window_left,
                             flashinfer::MaskMode::kNone, cuda_stream);
    }
  }

 private:
  // ---- Prefill path (query seq_len > 1) ---------------------------------

  void DispatchHeadDimPrefill(const Tensor& query, const Tensor& key,
                              const Tensor& value, Tensor& output,
                              int64_t num_heads, int64_t num_kv_heads,
                              int64_t head_size, double scale,
                              int64_t window_left,
                              flashinfer::MaskMode mask_mode,
                              typename Backend::Stream stream) const {
    switch (head_size) {
      case 64:
        DispatchMaskModePrefill<64>(query, key, value, output, num_heads,
                                    num_kv_heads, scale, window_left,
                                    mask_mode, stream);
        break;
      case 128:
        DispatchMaskModePrefill<128>(query, key, value, output, num_heads,
                                     num_kv_heads, scale, window_left,
                                     mask_mode, stream);
        break;
      case 256:
        DispatchMaskModePrefill<256>(query, key, value, output, num_heads,
                                     num_kv_heads, scale, window_left,
                                     mask_mode, stream);
        break;
      default:
        assert(false && "unsupported head dimension for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM>
  void DispatchMaskModePrefill(const Tensor& query, const Tensor& key,
                               const Tensor& value, Tensor& output,
                               int64_t num_heads, int64_t num_kv_heads,
                               double scale, int64_t window_left,
                               flashinfer::MaskMode mask_mode,
                               typename Backend::Stream stream) const {
    switch (mask_mode) {
      case flashinfer::MaskMode::kCausal:
        DispatchDtypePrefill<HEAD_DIM, flashinfer::MaskMode::kCausal>(
            query, key, value, output, num_heads, num_kv_heads, scale,
            window_left, stream);
        break;
      case flashinfer::MaskMode::kNone:
        DispatchDtypePrefill<HEAD_DIM, flashinfer::MaskMode::kNone>(
            query, key, value, output, num_heads, num_kv_heads, scale,
            window_left, stream);
        break;
      default:
        assert(false && "unsupported mask mode for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
  void DispatchDtypePrefill(const Tensor& query, const Tensor& key,
                            const Tensor& value, Tensor& output,
                            int64_t num_heads, int64_t num_kv_heads,
                            double scale, int64_t window_left,
                            typename Backend::Stream stream) const {
    DispatchFunc<Backend::kDeviceType, ReducedFloatTypes>(
        dtype_,
        [&](auto type_tag) {
          using DType = typename decltype(type_tag)::type;
          LaunchPrefill<HEAD_DIM, MASK_MODE, DType>(
              query, key, value, output, num_heads, num_kv_heads, scale,
              window_left, stream);
        },
        "CudaFlashAttention::prefill");
  }

  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename DType>
  void LaunchPrefill(const Tensor& query, const Tensor& key,
                     const Tensor& value, Tensor& output, int64_t num_heads,
                     int64_t num_kv_heads, double scale, int64_t window_left,
                     typename Backend::Stream stream) const {
    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;

    flashinfer::SinglePrefillParams<DType, DType, DType> params;
    params.q = reinterpret_cast<DType*>(const_cast<void*>(query.data()));
    params.k = reinterpret_cast<DType*>(const_cast<void*>(key.data()));
    params.v = reinterpret_cast<DType*>(const_cast<void*>(value.data()));
    params.o = reinterpret_cast<DType*>(output.data());
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.maybe_custom_mask = nullptr;

    params.qo_len = static_cast<uint32_t>(num_tokens_);
    params.kv_len = static_cast<uint32_t>(key.size(0));
    params.num_qo_heads = static_cast<uint32_t>(num_heads);
    params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    params.group_size = flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_heads / num_kv_heads));
    params.head_dim = HEAD_DIM;

    // Strides for NHD layout [seq_len, num_heads, head_dim].
    params.q_stride_n = static_cast<uint32_t>(num_heads * HEAD_DIM);
    params.q_stride_h = HEAD_DIM;
    params.k_stride_n = static_cast<uint32_t>(num_kv_heads * HEAD_DIM);
    params.k_stride_h = HEAD_DIM;
    params.v_stride_n = static_cast<uint32_t>(num_kv_heads * HEAD_DIM);
    params.v_stride_h = HEAD_DIM;

    params.sm_scale = static_cast<float>(scale);
    params.window_left = static_cast<int32_t>(window_left);
    params.logits_soft_cap = 0.0f;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.partition_kv = 0;

    cudaError_t err =
        flashinfer::SinglePrefillWithKVCacheDispatched<
            HEAD_DIM, HEAD_DIM, flashinfer::PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false, MASK_MODE, AttentionVariant>(
            params, /*tmp=*/nullptr, stream);

    assert(err == cudaSuccess &&
           "FlashInfer SinglePrefillWithKVCacheDispatched failed");
    (void)err;
  }

  // ---- Decode path (query seq_len == 1) ---------------------------------

  void DispatchHeadDimDecode(const Tensor& query, const Tensor& key,
                             const Tensor& value, Tensor& output,
                             int64_t num_heads, int64_t num_kv_heads,
                             int64_t head_size, double scale,
                             int64_t window_left,
                             typename Backend::Stream stream) const {
    switch (head_size) {
      case 64:
        DispatchDtypeDecode<64>(query, key, value, output, num_heads,
                                num_kv_heads, scale, window_left, stream);
        break;
      case 128:
        DispatchDtypeDecode<128>(query, key, value, output, num_heads,
                                 num_kv_heads, scale, window_left, stream);
        break;
      case 256:
        DispatchDtypeDecode<256>(query, key, value, output, num_heads,
                                 num_kv_heads, scale, window_left, stream);
        break;
      default:
        assert(false && "unsupported head dimension for FlashAttention decode");
    }
  }

  template <uint32_t HEAD_DIM>
  void DispatchDtypeDecode(const Tensor& query, const Tensor& key,
                           const Tensor& value, Tensor& output,
                           int64_t num_heads, int64_t num_kv_heads,
                           double scale, int64_t window_left,
                           typename Backend::Stream stream) const {
    DispatchFunc<Backend::kDeviceType, ReducedFloatTypes>(
        dtype_,
        [&](auto type_tag) {
          using DType = typename decltype(type_tag)::type;
          LaunchDecode<HEAD_DIM, DType>(query, key, value, output, num_heads,
                                        num_kv_heads, scale, window_left,
                                        stream);
        },
        "CudaFlashAttention::decode");
  }

  template <uint32_t HEAD_DIM, typename DType>
  void LaunchDecode(const Tensor& query, const Tensor& key,
                    const Tensor& value, Tensor& output, int64_t num_heads,
                    int64_t num_kv_heads, double scale, int64_t window_left,
                    typename Backend::Stream stream) const {
    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;

    uint32_t kv_len = static_cast<uint32_t>(key.size(0));

    flashinfer::SingleDecodeParams<DType, DType, DType> params(
        reinterpret_cast<DType*>(const_cast<void*>(query.data())),
        reinterpret_cast<DType*>(const_cast<void*>(key.data())),
        reinterpret_cast<DType*>(const_cast<void*>(value.data())),
        reinterpret_cast<DType*>(output.data()),
        /*maybe_alibi_slopes=*/nullptr, kv_len,
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
        flashinfer::QKVLayout::kNHD, HEAD_DIM,
        static_cast<int32_t>(window_left),
        /*logits_soft_cap=*/0.0f, static_cast<float>(scale),
        /*rope_scale=*/1.0f, /*rope_theta=*/1e4f);

    // Decode needs a temporary buffer for partial results.
    // Size: num_qo_heads * HEAD_DIM * sizeof(DType).
    // For single decode this is small enough to use nullptr (non-partitioned).
    cudaError_t err =
        flashinfer::SingleDecodeWithKVCacheDispatched<
            HEAD_DIM, flashinfer::PosEncodingMode::kNone, AttentionVariant>(
            params, /*tmp=*/nullptr, stream);

    assert(err == cudaSuccess &&
           "FlashInfer SingleDecodeWithKVCacheDispatched failed");
    (void)err;
  }

  // ---- Batch prefill (loop over sequences) --------------------------------

  void DispatchHeadDimBatchPrefill(
      const Tensor& query, const Tensor& key, const Tensor& value,
      const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_kv, Tensor& output,
      int64_t num_heads, int64_t num_kv_heads, int64_t head_size, double scale,
      int64_t window_left, flashinfer::MaskMode mask_mode,
      typename Backend::Stream stream) const {
    switch (head_size) {
      case 64:
        DispatchMaskModeBatchPrefill<64>(query, key, value, cu_seqlens_q,
                                         cu_seqlens_kv, output, num_heads,
                                         num_kv_heads, scale, window_left,
                                         mask_mode, stream);
        break;
      case 128:
        DispatchMaskModeBatchPrefill<128>(query, key, value, cu_seqlens_q,
                                          cu_seqlens_kv, output, num_heads,
                                          num_kv_heads, scale, window_left,
                                          mask_mode, stream);
        break;
      case 256:
        DispatchMaskModeBatchPrefill<256>(query, key, value, cu_seqlens_q,
                                          cu_seqlens_kv, output, num_heads,
                                          num_kv_heads, scale, window_left,
                                          mask_mode, stream);
        break;
      default:
        assert(false && "unsupported head dimension for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM>
  void DispatchMaskModeBatchPrefill(
      const Tensor& query, const Tensor& key, const Tensor& value,
      const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_kv, Tensor& output,
      int64_t num_heads, int64_t num_kv_heads, double scale,
      int64_t window_left, flashinfer::MaskMode mask_mode,
      typename Backend::Stream stream) const {
    switch (mask_mode) {
      case flashinfer::MaskMode::kCausal:
        DispatchDtypeBatchPrefill<HEAD_DIM, flashinfer::MaskMode::kCausal>(
            query, key, value, cu_seqlens_q, cu_seqlens_kv, output, num_heads,
            num_kv_heads, scale, window_left, stream);
        break;
      case flashinfer::MaskMode::kNone:
        DispatchDtypeBatchPrefill<HEAD_DIM, flashinfer::MaskMode::kNone>(
            query, key, value, cu_seqlens_q, cu_seqlens_kv, output, num_heads,
            num_kv_heads, scale, window_left, stream);
        break;
      default:
        assert(false && "unsupported mask mode for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
  void DispatchDtypeBatchPrefill(
      const Tensor& query, const Tensor& key, const Tensor& value,
      const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_kv, Tensor& output,
      int64_t num_heads, int64_t num_kv_heads, double scale,
      int64_t window_left, typename Backend::Stream stream) const {
    DispatchFunc<Backend::kDeviceType, ReducedFloatTypes>(
        dtype_,
        [&](auto type_tag) {
          using DType = typename decltype(type_tag)::type;
          LaunchBatchPrefill<HEAD_DIM, MASK_MODE, DType>(
              query, key, value, cu_seqlens_q, cu_seqlens_kv, output, num_heads,
              num_kv_heads, scale, window_left, stream);
        },
        "CudaFlashAttention::batch_prefill");
  }

  // Loop over packed sequences, calling SinglePrefill for each.
  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename DType>
  void LaunchBatchPrefill(const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor& cu_seqlens_q,
                          const Tensor& cu_seqlens_kv, Tensor& output,
                          int64_t num_heads, int64_t num_kv_heads, double scale,
                          int64_t window_left,
                          typename Backend::Stream stream) const {
    // Copy cu_seqlens from device to host.
    auto batch_size_plus_one = cu_seqlens_q.size(0);
    auto batch_size = batch_size_plus_one - 1;

    std::vector<int64_t> h_cu_q(batch_size_plus_one);
    std::vector<int64_t> h_cu_kv(batch_size_plus_one);
    cudaMemcpyAsync(h_cu_q.data(), cu_seqlens_q.data(),
                    batch_size_plus_one * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_cu_kv.data(), cu_seqlens_kv.data(),
                    batch_size_plus_one * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;

    auto* q_base = reinterpret_cast<DType*>(const_cast<void*>(query.data()));
    auto* k_base = reinterpret_cast<DType*>(const_cast<void*>(key.data()));
    auto* v_base = reinterpret_cast<DType*>(const_cast<void*>(value.data()));
    auto* o_base = reinterpret_cast<DType*>(output.data());

    uint32_t q_stride_n = static_cast<uint32_t>(num_heads * HEAD_DIM);
    uint32_t k_stride_n = static_cast<uint32_t>(num_kv_heads * HEAD_DIM);

    for (size_t i = 0; i < batch_size; ++i) {
      int64_t q_start = h_cu_q[i];
      int64_t kv_start = h_cu_kv[i];
      uint32_t qo_len = static_cast<uint32_t>(h_cu_q[i + 1] - q_start);
      uint32_t kv_len = static_cast<uint32_t>(h_cu_kv[i + 1] - kv_start);

      if (qo_len == 0) {
        continue;
      }

      flashinfer::SinglePrefillParams<DType, DType, DType> params;
      params.q = q_base + q_start * q_stride_n;
      params.k = k_base + kv_start * k_stride_n;
      params.v = v_base + kv_start * k_stride_n;
      params.o = o_base + q_start * q_stride_n;
      params.lse = nullptr;
      params.maybe_alibi_slopes = nullptr;
      params.maybe_custom_mask = nullptr;

      params.qo_len = qo_len;
      params.kv_len = kv_len;
      params.num_qo_heads = static_cast<uint32_t>(num_heads);
      params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
      params.group_size = flashinfer::uint_fastdiv(
          static_cast<uint32_t>(num_heads / num_kv_heads));
      params.head_dim = HEAD_DIM;

      params.q_stride_n = q_stride_n;
      params.q_stride_h = HEAD_DIM;
      params.k_stride_n = k_stride_n;
      params.k_stride_h = HEAD_DIM;
      params.v_stride_n = k_stride_n;
      params.v_stride_h = HEAD_DIM;

      params.sm_scale = static_cast<float>(scale);
      params.window_left = static_cast<int32_t>(window_left);
      params.logits_soft_cap = 0.0f;
      params.rope_rcp_scale = 1.0f;
      params.rope_rcp_theta = 1.0f;
      params.partition_kv = 0;

      cudaError_t err =
          flashinfer::SinglePrefillWithKVCacheDispatched<
              HEAD_DIM, HEAD_DIM, flashinfer::PosEncodingMode::kNone,
              /*USE_FP16_QK_REDUCTION=*/false, MASK_MODE, AttentionVariant>(
              params, /*tmp=*/nullptr, stream);

      assert(err == cudaSuccess &&
             "FlashInfer SinglePrefillWithKVCacheDispatched failed "
             "(batch prefill loop)");
      (void)err;
    }
  }

  // ---- Paged decode (loop over sequences) ---------------------------------

  void DispatchHeadDimPagedDecode(
      const Tensor& query, const Tensor& key, const Tensor& value,
      const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_kv,
      const Tensor& block_table, Tensor& output, int64_t num_heads,
      int64_t num_kv_heads, int64_t head_size, double scale,
      int64_t window_left, int64_t block_size,
      typename Backend::Stream stream) const {
    switch (head_size) {
      case 64:
        DispatchDtypePagedDecode<64>(query, key, value, cu_seqlens_q,
                                     cu_seqlens_kv, block_table, output,
                                     num_heads, num_kv_heads, scale,
                                     window_left, block_size, stream);
        break;
      case 128:
        DispatchDtypePagedDecode<128>(query, key, value, cu_seqlens_q,
                                      cu_seqlens_kv, block_table, output,
                                      num_heads, num_kv_heads, scale,
                                      window_left, block_size, stream);
        break;
      case 256:
        DispatchDtypePagedDecode<256>(query, key, value, cu_seqlens_q,
                                      cu_seqlens_kv, block_table, output,
                                      num_heads, num_kv_heads, scale,
                                      window_left, block_size, stream);
        break;
      default:
        assert(false &&
               "unsupported head dimension for FlashAttention paged decode");
    }
  }

  template <uint32_t HEAD_DIM>
  void DispatchDtypePagedDecode(
      const Tensor& query, const Tensor& key, const Tensor& value,
      const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_kv,
      const Tensor& block_table, Tensor& output, int64_t num_heads,
      int64_t num_kv_heads, double scale, int64_t window_left,
      int64_t block_size, typename Backend::Stream stream) const {
    DispatchFunc<Backend::kDeviceType, ReducedFloatTypes>(
        dtype_,
        [&](auto type_tag) {
          using DType = typename decltype(type_tag)::type;
          LaunchPagedDecode<HEAD_DIM, DType>(
              query, key, value, cu_seqlens_q, cu_seqlens_kv, block_table,
              output, num_heads, num_kv_heads, scale, window_left, block_size,
              stream);
        },
        "CudaFlashAttention::paged_decode");
  }

  // Loop over requests, gathering paged KV into a contiguous buffer and
  // calling SingleDecode for each.
  template <uint32_t HEAD_DIM, typename DType>
  void LaunchPagedDecode(const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_kv,
                         const Tensor& block_table, Tensor& output,
                         int64_t num_heads, int64_t num_kv_heads, double scale,
                         int64_t window_left, int64_t block_size,
                         typename Backend::Stream stream) const {
    // Copy metadata to host.
    auto num_reqs = block_table.size(0);
    auto max_blocks_per_req = block_table.size(1);

    // cu_seqlens are int64_t on device.
    std::vector<int64_t> h_cu_kv(num_reqs + 1);
    cudaMemcpyAsync(h_cu_kv.data(), cu_seqlens_kv.data(),
                    (num_reqs + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost,
                    stream);

    // block_table is int32 on device.
    std::vector<int32_t> h_block_table(num_reqs * max_blocks_per_req);
    cudaMemcpyAsync(h_block_table.data(), block_table.data(),
                    h_block_table.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;

    auto* q_base = reinterpret_cast<DType*>(const_cast<void*>(query.data()));
    auto* o_base = reinterpret_cast<DType*>(output.data());
    // KV cache has layout [num_blocks, block_size, num_kv_heads, head_dim].
    auto* kv_base = reinterpret_cast<DType*>(const_cast<void*>(key.data()));
    size_t page_stride =
        static_cast<size_t>(block_size) * num_kv_heads * HEAD_DIM;

    // Find the maximum KV length to size the temporary buffer.
    int64_t max_kv_len = 0;

    for (size_t i = 0; i < num_reqs; ++i) {
      int64_t kv_len = h_cu_kv[i + 1] - h_cu_kv[i];
      max_kv_len = std::max(max_kv_len, kv_len);
    }

    // Allocate a contiguous KV buffer for the longest sequence.
    size_t kv_buf_elems =
        static_cast<size_t>(max_kv_len) * num_kv_heads * HEAD_DIM;
    DType* d_k_buf = nullptr;
    DType* d_v_buf = nullptr;
    Backend::Malloc((void**)&d_k_buf, kv_buf_elems * sizeof(DType));
    Backend::Malloc((void**)&d_v_buf, kv_buf_elems * sizeof(DType));

    uint32_t q_stride_n = static_cast<uint32_t>(num_heads * HEAD_DIM);

    for (size_t i = 0; i < num_reqs; ++i) {
      int64_t kv_len = h_cu_kv[i + 1] - h_cu_kv[i];

      if (kv_len == 0) {
        continue;
      }

      // Gather KV pages into contiguous buffer.
      int64_t remaining = kv_len;
      size_t dst_offset = 0;
      size_t row_bytes = static_cast<size_t>(num_kv_heads) * HEAD_DIM *
                         sizeof(DType);

      for (size_t j = 0; j < max_blocks_per_req && remaining > 0; ++j) {
        int32_t block_idx = h_block_table[i * max_blocks_per_req + j];
        int64_t take = std::min(remaining, static_cast<int64_t>(block_size));
        size_t copy_bytes = static_cast<size_t>(take) * row_bytes;
        DType* src = kv_base + block_idx * page_stride;
        cudaMemcpyAsync(d_k_buf + dst_offset, src, copy_bytes,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_v_buf + dst_offset, src, copy_bytes,
                        cudaMemcpyDeviceToDevice, stream);
        dst_offset += take * num_kv_heads * HEAD_DIM;
        remaining -= take;
      }

      // Launch SingleDecode for this request.
      flashinfer::SingleDecodeParams<DType, DType, DType> params(
          q_base + i * q_stride_n, d_k_buf, d_v_buf,
          o_base + i * q_stride_n,
          /*maybe_alibi_slopes=*/nullptr,
          static_cast<uint32_t>(kv_len), static_cast<uint32_t>(num_heads),
          static_cast<uint32_t>(num_kv_heads), flashinfer::QKVLayout::kNHD,
          HEAD_DIM, static_cast<int32_t>(window_left),
          /*logits_soft_cap=*/0.0f, static_cast<float>(scale),
          /*rope_scale=*/1.0f, /*rope_theta=*/1e4f);

      cudaError_t err =
          flashinfer::SingleDecodeWithKVCacheDispatched<
              HEAD_DIM, flashinfer::PosEncodingMode::kNone, AttentionVariant>(
              params, /*tmp=*/nullptr, stream);

      assert(err == cudaSuccess &&
             "FlashInfer SingleDecodeWithKVCacheDispatched failed "
             "(paged decode loop)");
      (void)err;
    }

    Backend::Free(d_k_buf);
    Backend::Free(d_v_buf);
  }
};

}  // namespace infini::ops

#endif
