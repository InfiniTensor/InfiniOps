#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/_flash_attention_forward.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <tuple>

#include "torch/tensor_.h"

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 8>::operator()(
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
  const c10::cuda::CUDAGuard device_guard{device_index};

  const auto run = [&]() {
    auto at_q = ToAtenTensor<Device::Type::kNvidia>(const_cast<void*>(q.data()),
                                                    q_shape_, q_strides_,
                                                    q_dtype_, device_index_);
    auto at_k = ToAtenTensor<Device::Type::kNvidia>(const_cast<void*>(k.data()),
                                                    k_shape_, k_strides_,
                                                    k_dtype_, device_index_);
    auto at_v = ToAtenTensor<Device::Type::kNvidia>(const_cast<void*>(v.data()),
                                                    v_shape_, v_strides_,
                                                    v_dtype_, device_index_);
    auto at_cu_seqlens_q = ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
        cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
    auto at_cu_seqlens_k = ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
        cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
    auto at_out = ToAtenTensor<Device::Type::kNvidia>(
        out.data(), out_shape_, out_strides_, out_dtype_, device_index_);

    const std::optional<int64_t> window_size_left =
        window_size[0] < 0 ? std::nullopt
                           : std::optional<int64_t>{window_size[0]};
    const std::optional<int64_t> window_size_right =
        causal               ? std::optional<int64_t>{0}
        : window_size[1] < 0 ? std::nullopt
                             : std::optional<int64_t>{window_size[1]};

    auto result = at::_flash_attention_forward(
        at_q, at_k, at_v, at_cu_seqlens_q, at_cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, causal, false, softmax_scale, window_size_left,
        window_size_right, std::nullopt, std::nullopt);

    // ATen owns the returned tensor. Keep the InfiniOps trailing-output ABI by
    // copying it into the caller-provided buffer on the selected CUDA stream.
    at_out.copy_(std::get<0>(result));
  };

  const c10::cuda::CUDAStreamGuard stream_guard{
      c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_),
                                       device_index)};
  run();
}

}  // namespace infini::ops
