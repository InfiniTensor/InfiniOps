#include "torch/ops/flash_attn_with_kvcache/flash_attn.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "torch/tensor_.h"

namespace flash {

std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<const at::Tensor>& k, std::optional<const at::Tensor>& v,
    std::optional<const at::Tensor>& cache_seqlens,
    std::optional<const at::Tensor>& rotary_cos,
    std::optional<const at::Tensor>& rotary_sin,
    std::optional<const at::Tensor>& cache_batch_idx,
    std::optional<const at::Tensor>& cache_leftpad,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, std::optional<at::Tensor>& out,
    float softmax_scale, bool causal, int window_size_left,
    int window_size_right, float softcap, bool rotary_interleaved,
    int num_splits);

}  // namespace flash

namespace infini::ops {

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 8>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  Run(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
      std::nullopt, cache_batch_idx, cache_leftpad, block_table, softmax_scale,
      causal, window_size, softcap, rotary_interleaved, alibi_slopes,
      num_splits, out, softmax_lse);
  (void)return_softmax_lse;
}

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 8>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  Run(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, std::nullopt,
      cache_seqlens, cache_batch_idx, cache_leftpad, block_table, softmax_scale,
      causal, window_size, softcap, rotary_interleaved, alibi_slopes,
      num_splits, out, softmax_lse);
  (void)return_softmax_lse;
}

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 8>::Run(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<int64_t> scalar_cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const std::optional<Tensor> alibi_slopes,
    const int64_t num_splits, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  const auto device_index = static_cast<c10::DeviceIndex>(device_index_);
  const c10::cuda::CUDAGuard device_guard{device_index};
  const c10::cuda::CUDAStreamGuard stream_guard{
      c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_),
                                       device_index)};

  auto at_q =
      ToAtenTensor<Device::Type::kNvidia>(const_cast<void*>(q.data()), q_shape_,
                                          q_strides_, q_dtype_, device_index_);
  auto at_k_cache = ToAtenTensor<Device::Type::kNvidia>(
      k_cache.data(), k_cache_shape_, k_cache_strides_, k_cache_dtype_,
      device_index_);
  auto at_v_cache = ToAtenTensor<Device::Type::kNvidia>(
      v_cache.data(), v_cache_shape_, v_cache_strides_, v_cache_dtype_,
      device_index_);
  auto at_out = ToAtenTensor<Device::Type::kNvidia>(
      out.data(), out_shape_, out_strides_, out_dtype_, device_index_);
  std::optional<at::Tensor> at_softmax_lse;
  if (softmax_lse.has_value()) {
    at_softmax_lse.emplace(ToAtenTensor<Device::Type::kNvidia>(
        softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
        softmax_lse_dtype_, device_index_));
  }

  std::optional<const at::Tensor> at_k;
  std::optional<const at::Tensor> at_v;
  std::optional<const at::Tensor> at_rotary_cos;
  std::optional<const at::Tensor> at_rotary_sin;
  std::optional<const at::Tensor> at_cache_seqlens;
  std::optional<const at::Tensor> at_cache_batch_idx;
  std::optional<const at::Tensor> at_cache_leftpad;
  std::optional<at::Tensor> at_block_table;
  std::optional<at::Tensor> at_alibi_slopes;

  if (k.has_value()) {
    at_k.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(k->data()), k_shape_, k_strides_, k_dtype_,
        device_index_));
    at_v.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(v->data()), v_shape_, v_strides_, v_dtype_,
        device_index_));
  }
  if (rotary_cos.has_value()) {
    at_rotary_cos.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(rotary_cos->data()), rotary_cos_shape_,
        rotary_cos_strides_, rotary_cos_dtype_, device_index_));
    at_rotary_sin.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(rotary_sin->data()), rotary_sin_shape_,
        rotary_sin_strides_, rotary_sin_dtype_, device_index_));
  }
  if (cache_seqlens.has_value()) {
    at_cache_seqlens.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(cache_seqlens->data()), cache_seqlens_shape_,
        cache_seqlens_strides_, cache_seqlens_dtype_, device_index_));
  } else if (scalar_cache_seqlens.has_value()) {
    at_cache_seqlens.emplace(
        at::full({static_cast<int64_t>(batch_size_)}, *scalar_cache_seqlens,
                 at::TensorOptions().dtype(at::kInt).device(at_q.device())));
  }
  if (cache_batch_idx.has_value()) {
    at_cache_batch_idx.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(cache_batch_idx->data()), cache_batch_idx_shape_,
        cache_batch_idx_strides_, cache_batch_idx_dtype_, device_index_));
  }
  if (cache_leftpad.has_value()) {
    at_cache_leftpad.emplace(ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(cache_leftpad->data()), cache_leftpad_shape_,
        cache_leftpad_strides_, cache_leftpad_dtype_, device_index_));
  }
  if (block_table.has_value()) {
    at_block_table = ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(block_table->data()), block_table_shape_,
        block_table_strides_, block_table_dtype_, device_index_);
  }
  if (alibi_slopes.has_value()) {
    at_alibi_slopes = ToAtenTensor<Device::Type::kNvidia>(
        const_cast<void*>(alibi_slopes->data()), alibi_slopes_shape_,
        alibi_slopes_strides_, alibi_slopes_dtype_, device_index_);
  }

  std::optional<at::Tensor> at_out_optional;
  auto result = flash::mha_fwd_kvcache(
      at_q, at_k_cache, at_v_cache, at_k, at_v, at_cache_seqlens, at_rotary_cos,
      at_rotary_sin, at_cache_batch_idx, at_cache_leftpad, at_block_table,
      at_alibi_slopes, at_out_optional,
      static_cast<float>(softmax_scale.value_or(
          1.0 / std::sqrt(static_cast<double>(head_size_)))),
      causal, static_cast<int>(window_size[0]),
      static_cast<int>(window_size[1]), static_cast<float>(softcap),
      rotary_interleaved, static_cast<int>(num_splits));
  assert(!result.empty() && "`flash::mha_fwd_kvcache` returned no output");
  at_out.copy_(result[0]);
  if (at_softmax_lse.has_value()) {
    assert(result.size() >= 2 &&
           "`flash::mha_fwd_kvcache` did not return softmax LSE");
    at_softmax_lse->copy_(result[1]);
  }
}

}  // namespace infini::ops
