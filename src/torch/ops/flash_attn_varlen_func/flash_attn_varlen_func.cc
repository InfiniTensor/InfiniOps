#include "torch/ops/flash_attn_varlen_func/flash_attn_varlen_func.h"

#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/arange.h>
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
    Tensor out, std::optional<Tensor> softmax_lse,
    std::optional<Tensor> s_dmask) const {
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
    std::optional<at::Tensor> at_softmax_lse;
    std::optional<at::Tensor> at_s_dmask;
    if (softmax_lse.has_value()) {
      at_softmax_lse.emplace(ToAtenTensor<Device::Type::kNvidia>(
          softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
          softmax_lse_dtype_, device_index_));
      at_s_dmask.emplace(ToAtenTensor<Device::Type::kNvidia>(
          s_dmask->data(), s_dmask_shape_, s_dmask_strides_, s_dmask_dtype_,
          device_index_));
    }

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

    // ATen owns the returned tensors. Keep the InfiniOps trailing-output ABI
    // by copying them into caller-provided buffers on the selected CUDA stream.
    at_out.copy_(std::get<0>(result));
    if (at_softmax_lse.has_value()) {
      const auto& result_softmax_lse = std::get<1>(result);
      if (result_softmax_lse.dim() == 3) {
        // ATen may return padded (batch, heads, max_q) storage instead of the
        // packed FlashAttention (heads, total_q) layout.
        const auto batch_size = result_softmax_lse.size(0);
        const auto q_lengths = at_cu_seqlens_q.narrow(0, 1, batch_size) -
                               at_cu_seqlens_q.narrow(0, 0, batch_size);
        const auto positions =
            at::arange(result_softmax_lse.size(2), at_cu_seqlens_q.options());
        const auto valid_positions =
            positions.unsqueeze(0).lt(q_lengths.unsqueeze(1));
        const auto packed_softmax_lse =
            result_softmax_lse.transpose(1, 2)
                .masked_select(valid_positions.unsqueeze(2))
                .view({at_q.size(0), at_q.size(1)})
                .transpose(0, 1);
        at_softmax_lse->copy_(packed_softmax_lse);
      } else {
        at_softmax_lse->copy_(result_softmax_lse);
      }
      at_s_dmask->copy_(std::get<4>(result));
    }
  };

  const c10::cuda::CUDAStreamGuard stream_guard{
      c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_),
                                       device_index)};
  run();
}

}  // namespace infini::ops
