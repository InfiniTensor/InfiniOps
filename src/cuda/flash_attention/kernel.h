#ifndef INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_
#define INFINI_OPS_CUDA_FLASH_ATTENTION_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/flash_attention.h"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/pos_enc.cuh"

namespace infini::ops {

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

    if (causal) {
      DispatchHeadDim(query, key, value, output, num_heads, num_kv_heads,
                      head_size, scale, window_left,
                      flashinfer::MaskMode::kCausal, cuda_stream);
    } else {
      DispatchHeadDim(query, key, value, output, num_heads, num_kv_heads,
                      head_size, scale, window_left,
                      flashinfer::MaskMode::kNone, cuda_stream);
    }
  }

 private:
  void DispatchHeadDim(const Tensor& query, const Tensor& key,
                       const Tensor& value, Tensor& output, int64_t num_heads,
                       int64_t num_kv_heads, int64_t head_size, double scale,
                       int64_t window_left, flashinfer::MaskMode mask_mode,
                       typename Backend::Stream stream) const {
    switch (head_size) {
      case 64:
        DispatchMaskMode<64>(query, key, value, output, num_heads, num_kv_heads,
                             scale, window_left, mask_mode, stream);
        break;
      case 128:
        DispatchMaskMode<128>(query, key, value, output, num_heads,
                              num_kv_heads, scale, window_left, mask_mode,
                              stream);
        break;
      case 256:
        DispatchMaskMode<256>(query, key, value, output, num_heads,
                              num_kv_heads, scale, window_left, mask_mode,
                              stream);
        break;
      default:
        assert(false && "unsupported head dimension for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM>
  void DispatchMaskMode(const Tensor& query, const Tensor& key,
                        const Tensor& value, Tensor& output, int64_t num_heads,
                        int64_t num_kv_heads, double scale, int64_t window_left,
                        flashinfer::MaskMode mask_mode,
                        typename Backend::Stream stream) const {
    switch (mask_mode) {
      case flashinfer::MaskMode::kCausal:
        DispatchDtype<HEAD_DIM, flashinfer::MaskMode::kCausal>(
            query, key, value, output, num_heads, num_kv_heads, scale,
            window_left, stream);
        break;
      case flashinfer::MaskMode::kNone:
        DispatchDtype<HEAD_DIM, flashinfer::MaskMode::kNone>(
            query, key, value, output, num_heads, num_kv_heads, scale,
            window_left, stream);
        break;
      default:
        assert(false && "unsupported mask mode for FlashAttention");
    }
  }

  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
  void DispatchDtype(const Tensor& query, const Tensor& key,
                     const Tensor& value, Tensor& output, int64_t num_heads,
                     int64_t num_kv_heads, double scale, int64_t window_left,
                     typename Backend::Stream stream) const {
    DispatchFunc<Backend::kDeviceType, ReducedFloatTypes>(
        dtype_,
        [&](auto type_tag) {
          using DType = typename decltype(type_tag)::type;
          LaunchKernel<HEAD_DIM, MASK_MODE, DType>(query, key, value, output,
                                                   num_heads, num_kv_heads,
                                                   scale, window_left, stream);
        },
        "CudaFlashAttention::operator()");
  }

  template <uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename DType>
  void LaunchKernel(const Tensor& query, const Tensor& key,
                    const Tensor& value, Tensor& output, int64_t num_heads,
                    int64_t num_kv_heads, double scale, int64_t window_left,
                    typename Backend::Stream stream) const {
    // Determine whether sliding window is active.
    constexpr bool kUseSlidingWindow = false;

    using AttentionVariant =
        flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                     kUseSlidingWindow,
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

    // For non-partitioned KV, tmp buffer is not needed.
    cudaError_t err =
        flashinfer::SinglePrefillWithKVCacheDispatched<
            HEAD_DIM, HEAD_DIM, flashinfer::PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false, MASK_MODE, AttentionVariant>(
            params, /*tmp=*/nullptr, stream);

    assert(err == cudaSuccess &&
           "FlashInfer SinglePrefillWithKVCacheDispatched failed");
    (void)err;
  }
};

}  // namespace infini::ops

#endif
