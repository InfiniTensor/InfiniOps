#include "mla_kernel.h"

#include <cuda_runtime.h>

#include <cmath>

#include "native/cuda/nvidia/ops/flash_mla_launch.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

struct Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 1>::Impl {
  native_flash_mla::MlaDecodeLaunch params{};
  std::optional<double> scale;
  std::array<int64_t, 2> requested_window;
  bool lengths_present;
};

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 1>::prepare(
    const Tensor& q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor>& lengths, const std::optional<Tensor>& blocks,
    const std::optional<double>& scale,
    const std::vector<int64_t>& window_size) {
  auto state = std::make_shared<Impl>();
  state->scale = scale;
  state->requested_window = {window_size[0], window_size[1]};
  state->lengths_present = lengths.has_value();
  auto& p = state->params;
  p.num_heads = q.size(2);
  p.num_seqs = q.size(0);
  p.num_kv_heads = k_cache.size(2);
  p.max_num_blocks_per_seq = blocks->size(1);
  p.block_size = k_cache.size(1);
  p.scale = static_cast<float>(
      scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(3)))));
  if (!std::isfinite(p.scale))
    throw std::invalid_argument("nonfinite native MLA KV Attention scale");
  p.block_table_batch_stride = blocks->stride(0);
  p.q_batch_stride = q.stride(0);
  p.q_head_stride = q.stride(2);
  p.k_block_stride = k_cache.stride(0);
  p.k_head_stride = k_cache.stride(2);
  p.k_slot_stride = k_cache.stride(1);
  p.v_block_stride = v_cache.stride(0);
  p.v_head_stride = v_cache.stride(2);
  p.v_slot_stride = v_cache.stride(1);
  p.o_batch_stride = out_strides_[0];
  p.o_head_stride = out_strides_[2];
  p.seqlens_stride = lengths ? lengths->stride(0) : 0;
  impl_ = std::move(state);
}

Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 1>::~Operator() {
  impl_.reset();
}

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 1>::execute(
    const Tensor& q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor>& lengths, const std::optional<Tensor>& blocks,
    const std::optional<double>& scale, const std::vector<int64_t>& window_size,
    Tensor out) const {
  if (q.shape() != q_shape_ || q.strides() != q_strides_ ||
      k_cache.shape() != k_cache_shape_ ||
      k_cache.strides() != k_cache_strides_ ||
      v_cache.shape() != v_cache_shape_ ||
      v_cache.strides() != v_cache_strides_ || out.shape() != out_shape_ ||
      out.strides() != out_strides_ ||
      (blocks ? Tensor::Shape{blocks->shape()} : Tensor::Shape{}) !=
          block_table_shape_ ||
      (blocks ? Tensor::Strides{blocks->strides()} : Tensor::Strides{}) !=
          block_table_strides_ ||
      scale != impl_->scale || window_size.size() != 2 ||
      window_size[0] != impl_->requested_window[0] ||
      window_size[1] != impl_->requested_window[1] ||
      lengths.has_value() != impl_->lengths_present)
    throw std::invalid_argument(
        "native MLA KV Attention prepared metadata changed");
  auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
  native_flash_mla::launch_mla_decode(
      out.data(), q.data(), k_cache.data(), v_cache.data(), blocks->data(),
      lengths->data(), q_dtype_, block_table_dtype_, impl_->params, stream);
}

}  // namespace infini::ops
