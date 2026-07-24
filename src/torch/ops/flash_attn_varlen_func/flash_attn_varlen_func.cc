#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/equal.h>
#if defined(WITH_NVIDIA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#elif defined(WITH_MOORE)
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#endif

#include <tuple>

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void AtenFlashAttnVarlenFunc<kDev>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const std::optional<Tensor> alibi_slopes, const bool deterministic,
    const bool return_attn_probs, const std::optional<Tensor> block_table,
    Tensor out) const {
  (void)softcap;
  (void)alibi_slopes;
  (void)deterministic;
  (void)return_attn_probs;
  (void)block_table;

  const auto device_index = static_cast<c10::DeviceIndex>(device_index_);
  const auto run = [&]() {
    auto at_q = ToAtenTensor<kDev>(const_cast<void*>(q.data()), q_shape_,
                                   q_strides_, q_dtype_, device_index_);
    auto at_k = ToAtenTensor<kDev>(const_cast<void*>(k.data()), k_shape_,
                                   k_strides_, k_dtype_, device_index_);
    auto at_v = ToAtenTensor<kDev>(const_cast<void*>(v.data()), v_shape_,
                                   v_strides_, v_dtype_, device_index_);
    auto at_cu_seqlens_q = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<kDev>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                     out_dtype_, device_index_);

    if constexpr (kDev == Device::Type::kMoore) {
      TORCH_CHECK(!causal || at::equal(at_cu_seqlens_q, at_cu_seqlens_k),
                  "TorchMusa FlashAttention requires matching query and key "
                  "sequence lengths for causal attention");
    }

    std::optional<int64_t> window_size_left;
    std::optional<int64_t> window_size_right;
    if constexpr (kDev == Device::Type::kNvidia) {
      if (window_size[0] >= 0) {
        window_size_left = window_size[0];
      }
      if (causal) {
        window_size_right = 0;
      } else if (window_size[1] >= 0) {
        window_size_right = window_size[1];
      }
    }

    auto result = at::_flash_attention_forward(
        at_q, at_k, at_v, at_cu_seqlens_q, at_cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, causal, false, softmax_scale, window_size_left,
        window_size_right, std::nullopt, std::nullopt);

    // ATen owns the returned tensor. Keep the InfiniOps trailing-output ABI by
    // copying it into the caller-provided buffer on the selected stream.
    at_out.copy_(std::get<0>(result));
  };

#if defined(WITH_NVIDIA)
  static_assert(kDev == Device::Type::kNvidia);
  const c10::cuda::CUDAGuard device_guard{device_index};
  const c10::cuda::CUDAStreamGuard stream_guard{
      c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_),
                                       device_index)};
#elif defined(WITH_MOORE)
  static_assert(kDev == Device::Type::kMoore);
  const at::musa::MUSAStreamGuard stream_guard{at::musa::getStreamFromExternal(
      reinterpret_cast<musaStream_t>(stream_), device_index)};
#endif
  run();
}

#if defined(WITH_NVIDIA)
template class AtenFlashAttnVarlenFunc<Device::Type::kNvidia>;
#elif defined(WITH_MOORE)
template class AtenFlashAttnVarlenFunc<Device::Type::kMoore>;
#endif

}  // namespace infini::ops
