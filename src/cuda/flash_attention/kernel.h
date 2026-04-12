#ifndef INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "base/flash_attention.h"
#include "flashinfer/allocator.h"
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/page.cuh"
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
// Batch prefill uses `BatchPrefillWithRaggedKVCacheDispatched` with the
// `PrefillPlan` scheduler (split-KV disabled).  Paged decode uses
// `BatchDecodeWithPagedKVCacheDispatched` with the `DecodePlan` scheduler.
template <typename Backend>
class CudaFlashAttention : public FlashAttention {
  // FlashInfer recommends 128 MB for each scheduler workspace buffer.
  static constexpr size_t kIntWorkspaceBytes = 128 * 1024 * 1024;
  static constexpr size_t kFloatWorkspaceBytes = 128 * 1024 * 1024;

  // Scratch region after the two large buffers, used for small metadata
  // arrays (`d_qo_indptr`, `d_kv_indptr`, page indices, etc.).
  static constexpr size_t kScratchBytes = 8 * 1024 * 1024;  // 8 MB.

  // Pinned host staging buffer for FlashInfer scheduler.
  static constexpr size_t kPinnedBytes = kIntWorkspaceBytes;

 public:
  template <typename... Args>
  CudaFlashAttention(Args&&... args) : FlashAttention(std::forward<Args>(args)...) {
    cudaMalloc(&default_workspace_, workspace_size_in_bytes());
    assert(default_workspace_ && "failed to allocate device workspace");
    cudaMallocHost(&pinned_workspace_, kPinnedBytes);
    assert(pinned_workspace_ && "failed to allocate pinned host workspace");
  }

  ~CudaFlashAttention() override {
    if (default_workspace_) {
      cudaFree(default_workspace_);
      default_workspace_ = nullptr;
    }

    if (pinned_workspace_) {
      cudaFreeHost(pinned_workspace_);
      pinned_workspace_ = nullptr;
    }
  }

  // Non-copyable, non-movable (pinned memory ownership).
  CudaFlashAttention(const CudaFlashAttention&) = delete;
  CudaFlashAttention& operator=(const CudaFlashAttention&) = delete;

  std::size_t workspace_size_in_bytes() const override {
    // int_workspace (128 MB) + float_workspace (128 MB) + scratch (8 MB).
    return kIntWorkspaceBytes + kFloatWorkspaceBytes + kScratchBytes;
  }

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

  // Batch prefill using FlashInfer's native batch kernel with scheduler.
  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename DType>
  void LaunchBatchPrefill(const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor& cu_seqlens_q,
                          const Tensor& cu_seqlens_kv, Tensor& output,
                          int64_t num_heads, int64_t num_kv_heads, double scale,
                          int64_t window_left,
                          typename Backend::Stream stream) const {
    // Copy cu_seqlens (int64) from device to host, then narrow to int32.
    auto batch_size_plus_one = cu_seqlens_q.size(0);
    auto batch_size = static_cast<uint32_t>(batch_size_plus_one - 1);

    std::vector<int64_t> h_cu_q_i64(batch_size_plus_one);
    std::vector<int64_t> h_cu_kv_i64(batch_size_plus_one);
    cudaMemcpyAsync(h_cu_q_i64.data(), cu_seqlens_q.data(),
                    batch_size_plus_one * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_cu_kv_i64.data(), cu_seqlens_kv.data(),
                    batch_size_plus_one * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Convert to int32 for FlashInfer scheduler (IdType = int32_t).
    std::vector<int32_t> h_cu_q(batch_size_plus_one);
    std::vector<int32_t> h_cu_kv(batch_size_plus_one);

    for (size_t i = 0; i < batch_size_plus_one; ++i) {
      h_cu_q[i] = static_cast<int32_t>(h_cu_q_i64[i]);
      h_cu_kv[i] = static_cast<int32_t>(h_cu_kv_i64[i]);
    }

    uint32_t total_num_rows = static_cast<uint32_t>(h_cu_q[batch_size]);

    // Partition pre-allocated device workspace into sub-regions.
    void* active_workspace = workspace_ ? workspace_ : default_workspace_;
    size_t active_workspace_size = workspace_ ? workspace_size_in_bytes_
                                              : workspace_size_in_bytes();
    char* ws = static_cast<char*>(active_workspace);
    size_t ws_offset = 0;

    void* int_buf = ws + ws_offset;
    ws_offset += kIntWorkspaceBytes;

    // Run PrefillPlan with split-KV disabled for simplicity.
    flashinfer::PrefillPlanInfo plan_info;
    cudaError_t plan_err = flashinfer::PrefillPlan<int32_t>(
        /*float_buffer=*/nullptr,
        /*float_workspace_size_in_bytes=*/0, int_buf, pinned_workspace_,
        kIntWorkspaceBytes, plan_info, h_cu_q.data(), h_cu_kv.data(),
        total_num_rows, batch_size,
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
        /*head_dim_qk=*/HEAD_DIM, /*head_dim_vo=*/HEAD_DIM,
        /*page_size=*/1,
        /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(DType),
        static_cast<int32_t>(window_left),
        /*fixed_split_size=*/0, /*disable_split_kv=*/true,
        /*num_colocated_ctas=*/0, stream);

    assert(plan_err == cudaSuccess && "FlashInfer PrefillPlan failed");
    (void)plan_err;

    // Upload cu_seqlens as int32 to device from the scratch region.
    // Skip float workspace region (unused for prefill) to reach scratch.
    ws_offset += kFloatWorkspaceBytes;

    int32_t* d_qo_indptr = reinterpret_cast<int32_t*>(ws + ws_offset);
    ws_offset += batch_size_plus_one * sizeof(int32_t);

    int32_t* d_kv_indptr = reinterpret_cast<int32_t*>(ws + ws_offset);
    ws_offset += batch_size_plus_one * sizeof(int32_t);

    assert(ws_offset <= active_workspace_size &&
           "FlashAttention batch prefill workspace overflow");

    cudaMemcpyAsync(d_qo_indptr, h_cu_q.data(),
                    batch_size_plus_one * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kv_indptr, h_cu_kv.data(),
                    batch_size_plus_one * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;
    using Params =
        flashinfer::BatchPrefillRaggedParams<DType, DType, DType, int32_t>;

    uint32_t q_stride_n = static_cast<uint32_t>(num_heads * HEAD_DIM);
    uint32_t k_stride_n = static_cast<uint32_t>(num_kv_heads * HEAD_DIM);

    Params params;
    params.q =
        reinterpret_cast<DType*>(const_cast<void*>(query.data()));
    params.k =
        reinterpret_cast<DType*>(const_cast<void*>(key.data()));
    params.v =
        reinterpret_cast<DType*>(const_cast<void*>(value.data()));
    params.o = reinterpret_cast<DType*>(output.data());
    params.lse = nullptr;
    params.maybe_custom_mask = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.maybe_k_rope_offset = nullptr;
    params.maybe_mask_indptr = nullptr;
    params.q_indptr = d_qo_indptr;
    params.kv_indptr = d_kv_indptr;
    params.num_qo_heads = static_cast<uint32_t>(num_heads);
    params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    params.group_size = flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_heads / num_kv_heads));
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

    // Fill scheduling metadata from plan_info.
    params.padded_batch_size =
        static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv = plan_info.split_kv;
    params.max_total_num_rows = total_num_rows;
    params.total_num_rows = plan_info.enable_cuda_graph
        ? flashinfer::GetPtrFromBaseOffset<uint32_t>(
              int_buf, plan_info.total_num_rows_offset)
        : nullptr;
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.kv_tile_indices_offset);
    params.merge_indptr = plan_info.split_kv
        ? flashinfer::GetPtrFromBaseOffset<int32_t>(
              int_buf, plan_info.merge_indptr_offset)
        : nullptr;
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.block_valid_mask = plan_info.split_kv
        ? flashinfer::GetPtrFromBaseOffset<bool>(
              int_buf, plan_info.block_valid_mask_offset)
        : nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    // Dispatch on CTA_TILE_Q determined by the plan.
    uint32_t cta_tile_q = static_cast<uint32_t>(plan_info.cta_tile_q);

    switch (cta_tile_q) {
      case 128:
        LaunchBatchPrefillKernel<128, HEAD_DIM, MASK_MODE, DType,
                                 AttentionVariant>(params, stream);
        break;
      case 64:
        LaunchBatchPrefillKernel<64, HEAD_DIM, MASK_MODE, DType,
                                 AttentionVariant>(params, stream);
        break;
      case 16:
        LaunchBatchPrefillKernel<16, HEAD_DIM, MASK_MODE, DType,
                                 AttentionVariant>(params, stream);
        break;
      default:
        assert(false && "unsupported CTA_TILE_Q from PrefillPlan");
    }

  }

  // Helper to dispatch batch prefill kernel with a compile-time CTA_TILE_Q.
  template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_VAL,
            flashinfer::MaskMode MASK_MODE_VAL, typename DType,
            typename AttentionVariant>
  static void LaunchBatchPrefillKernel(
      flashinfer::BatchPrefillRaggedParams<DType, DType, DType, int32_t>&
          params,
      typename Backend::Stream stream) {
    cudaError_t err =
        flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM_VAL, HEAD_DIM_VAL,
            flashinfer::PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false, MASK_MODE_VAL,
            AttentionVariant>(params, /*tmp_v=*/nullptr,
                              /*tmp_s=*/nullptr,
                              /*enable_pdl=*/false, stream);

    assert(err == cudaSuccess &&
           "FlashInfer BatchPrefillWithRaggedKVCacheDispatched failed");
    (void)err;
  }

  // ---- Paged decode (batch via scheduler) ----------------------------------

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

  // Batch paged decode using FlashInfer's native batch kernel with scheduler.
  template <uint32_t HEAD_DIM, typename DType>
  void LaunchPagedDecode(const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& cu_seqlens_q,
                         const Tensor& cu_seqlens_kv,
                         const Tensor& block_table, Tensor& output,
                         int64_t num_heads, int64_t num_kv_heads, double scale,
                         int64_t window_left, int64_t block_size,
                         typename Backend::Stream stream) const {
    // Copy metadata to host.
    auto num_reqs = static_cast<uint32_t>(block_table.size(0));
    auto max_blocks_per_req = block_table.size(1);

    // cu_seqlens_kv is int64 on device.
    std::vector<int64_t> h_cu_kv_i64(num_reqs + 1);
    cudaMemcpyAsync(h_cu_kv_i64.data(), cu_seqlens_kv.data(),
                    (num_reqs + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // Build page indptr and last_page_len arrays for paged_kv_t.
    // block_table has shape [num_reqs, max_blocks_per_req] on device.
    std::vector<int32_t> h_page_indptr(num_reqs + 1);
    std::vector<int32_t> h_last_page_len(num_reqs);
    h_page_indptr[0] = 0;

    for (uint32_t i = 0; i < num_reqs; ++i) {
      int64_t kv_len = h_cu_kv_i64[i + 1] - h_cu_kv_i64[i];
      uint32_t num_pages =
          kv_len > 0
              ? static_cast<uint32_t>((kv_len + block_size - 1) / block_size)
              : 0;
      h_page_indptr[i + 1] = h_page_indptr[i] + static_cast<int32_t>(num_pages);

      if (kv_len > 0) {
        int32_t last_len = static_cast<int32_t>(kv_len % block_size);
        h_last_page_len[i] = last_len == 0
            ? static_cast<int32_t>(block_size)
            : last_len;
      } else {
        h_last_page_len[i] = 0;
      }
    }

    int32_t total_pages = h_page_indptr[num_reqs];

    // Flatten block_table into a contiguous page indices array on device.
    // block_table is [num_reqs, max_blocks_per_req] int32 on device; we need
    // a flat [total_pages] array with only the valid entries.
    std::vector<int32_t> h_block_table(num_reqs * max_blocks_per_req);
    cudaMemcpyAsync(h_block_table.data(), block_table.data(),
                    h_block_table.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int32_t> h_page_indices(total_pages);
    int32_t idx = 0;

    for (uint32_t i = 0; i < num_reqs; ++i) {
      int32_t num_pages =
          h_page_indptr[i + 1] - h_page_indptr[i];

      for (int32_t j = 0; j < num_pages; ++j) {
        h_page_indices[idx++] =
            h_block_table[i * max_blocks_per_req + j];
      }
    }

    // Partition pre-allocated device workspace into sub-regions.
    void* active_workspace = workspace_ ? workspace_ : default_workspace_;
    size_t active_workspace_size = workspace_ ? workspace_size_in_bytes_
                                              : workspace_size_in_bytes();
    char* ws = static_cast<char*>(active_workspace);
    size_t ws_offset = 0;

    void* int_buf = ws + ws_offset;
    ws_offset += kIntWorkspaceBytes;

    void* float_buf = ws + ws_offset;
    ws_offset += kFloatWorkspaceBytes;

    // Small metadata arrays from the scratch region.
    int32_t* d_page_indices = reinterpret_cast<int32_t*>(ws + ws_offset);
    ws_offset += std::max<size_t>(total_pages, 1) * sizeof(int32_t);

    int32_t* d_page_indptr = reinterpret_cast<int32_t*>(ws + ws_offset);
    ws_offset += (num_reqs + 1) * sizeof(int32_t);

    int32_t* d_last_page_len = reinterpret_cast<int32_t*>(ws + ws_offset);
    ws_offset += num_reqs * sizeof(int32_t);

    assert(ws_offset <= active_workspace_size &&
           "FlashAttention paged decode workspace overflow");

    if (total_pages > 0) {
      cudaMemcpyAsync(d_page_indices, h_page_indices.data(),
                      total_pages * sizeof(int32_t), cudaMemcpyHostToDevice,
                      stream);
    }

    cudaMemcpyAsync(d_page_indptr, h_page_indptr.data(),
                    (num_reqs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_last_page_len, h_last_page_len.data(),
                    num_reqs * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);

    // KV cache layout: [num_blocks, block_size, num_kv_heads, head_dim] (NHD).
    auto* kv_data =
        reinterpret_cast<DType*>(const_cast<void*>(key.data()));

    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(block_size), HEAD_DIM, num_reqs,
        flashinfer::QKVLayout::kNHD, kv_data, kv_data, d_page_indices,
        d_page_indptr, d_last_page_len);

    // Device workspace was partitioned above; use pinned host member.

    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     /*use_sliding_window=*/false,
                                     /*use_logits_soft_cap=*/false,
                                     /*use_alibi=*/false>;
    using Params =
        flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;

    uint32_t group_size = static_cast<uint32_t>(num_heads / num_kv_heads);

    // Dispatch on GQA group size for DecodePlan + kernel launch. The group
    // size must be a compile-time constant for the work estimation function.
    switch (group_size) {
      case 1:
        LaunchPagedDecodeInner<HEAD_DIM, 1, DType, AttentionVariant, Params>(
            query, output, paged_kv, float_buf, kFloatWorkspaceBytes, int_buf,
            pinned_workspace_, kIntWorkspaceBytes, h_page_indptr.data(), num_reqs,
            num_heads, scale, window_left, block_size, stream);
        break;
      case 2:
        LaunchPagedDecodeInner<HEAD_DIM, 2, DType, AttentionVariant, Params>(
            query, output, paged_kv, float_buf, kFloatWorkspaceBytes, int_buf,
            pinned_workspace_, kIntWorkspaceBytes, h_page_indptr.data(), num_reqs,
            num_heads, scale, window_left, block_size, stream);
        break;
      case 4:
        LaunchPagedDecodeInner<HEAD_DIM, 4, DType, AttentionVariant, Params>(
            query, output, paged_kv, float_buf, kFloatWorkspaceBytes, int_buf,
            pinned_workspace_, kIntWorkspaceBytes, h_page_indptr.data(), num_reqs,
            num_heads, scale, window_left, block_size, stream);
        break;
      case 8:
        LaunchPagedDecodeInner<HEAD_DIM, 8, DType, AttentionVariant, Params>(
            query, output, paged_kv, float_buf, kFloatWorkspaceBytes, int_buf,
            pinned_workspace_, kIntWorkspaceBytes, h_page_indptr.data(), num_reqs,
            num_heads, scale, window_left, block_size, stream);
        break;
      default:
        assert(false && "unsupported GQA group size for paged decode");
    }

  }

  // Inner helper for paged decode, templated on compile-time GROUP_SIZE.
  template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE, typename DType,
            typename AttentionVariant, typename Params>
  static void LaunchPagedDecodeInner(
      const Tensor& query, Tensor& output,
      flashinfer::paged_kv_t<DType, int32_t>& paged_kv, void* float_buf,
      size_t float_ws, void* int_buf, void* pinned_buf, size_t int_ws,
      int32_t* page_indptr_h, uint32_t num_reqs, int64_t num_heads,
      double scale, int64_t window_left, int64_t block_size,
      typename Backend::Stream stream) {
    // Work estimation function with compile-time GROUP_SIZE.
    cudaError_t (*work_estimation_func)(
        bool&, uint32_t&, uint32_t&, uint32_t&, uint32_t&, uint32_t,
        int32_t*, uint32_t, uint32_t, bool, cudaStream_t) =
        flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, HEAD_DIM, flashinfer::PosEncodingMode::kNone,
            AttentionVariant, Params>;

    flashinfer::DecodePlanInfo plan_info;
    cudaError_t plan_err = flashinfer::DecodePlan<
        HEAD_DIM, flashinfer::PosEncodingMode::kNone, AttentionVariant,
        Params>(
        float_buf, float_ws, int_buf, pinned_buf, int_ws, plan_info,
        page_indptr_h, num_reqs, static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(block_size),
        /*enable_cuda_graph=*/false, stream, work_estimation_func);

    assert(plan_err == cudaSuccess && "FlashInfer DecodePlan failed");
    (void)plan_err;

    // Fill BatchDecodeParams.
    uint32_t q_stride_n = static_cast<uint32_t>(num_heads * HEAD_DIM);

    Params params(
        reinterpret_cast<DType*>(const_cast<void*>(query.data())),
        /*q_rope_offset=*/nullptr, paged_kv,
        reinterpret_cast<DType*>(output.data()),
        /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr,
        static_cast<uint32_t>(num_heads),
        static_cast<int32_t>(q_stride_n),
        static_cast<int32_t>(HEAD_DIM),
        static_cast<int32_t>(window_left),
        /*logits_soft_cap=*/0.0f, static_cast<float>(scale),
        /*rope_scale=*/1.0f, /*rope_theta=*/1e4f);

    // Fill scheduling metadata from plan_info.
    params.padded_batch_size =
        static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv = plan_info.split_kv;
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.request_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.block_valid_mask = plan_info.split_kv
        ? flashinfer::GetPtrFromBaseOffset<bool>(
              int_buf, plan_info.block_valid_mask_offset)
        : nullptr;

    // Temporary buffers for split-KV reduction.
    DType* tmp_v = plan_info.split_kv
        ? flashinfer::GetPtrFromBaseOffset<DType>(
              float_buf, plan_info.v_offset)
        : nullptr;
    float* tmp_s = plan_info.split_kv
        ? flashinfer::GetPtrFromBaseOffset<float>(
              float_buf, plan_info.s_offset)
        : nullptr;

    cudaError_t err =
        flashinfer::BatchDecodeWithPagedKVCacheDispatched<
            HEAD_DIM, flashinfer::PosEncodingMode::kNone, AttentionVariant>(
            params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);

    assert(err == cudaSuccess &&
           "FlashInfer BatchDecodeWithPagedKVCacheDispatched failed");
    (void)err;
  }

  // Device workspace, allocated once in the constructor. Used as fallback
  // when the handle does not provide a workspace buffer.
  mutable void* default_workspace_{nullptr};

  // Pinned host staging buffer, allocated once in the constructor.
  mutable void* pinned_workspace_{nullptr};
};

}  // namespace infini::ops

#endif
