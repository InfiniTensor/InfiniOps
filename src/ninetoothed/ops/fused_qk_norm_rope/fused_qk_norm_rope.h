#ifndef INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_ROPE_H_
#define INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_ROPE_H_

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "base/fused_qk_norm_rope.h"
#include "cuda_runtime_api.h"
#include "data_type.h"
#include "fused_qk_norm_rope/infini_ops_ninetoothed_fused_qk_norm_rope.h"
#include "ninetoothed/tensor.h"

namespace infini::ops {

template <>
class Operator<FusedQkNormRope, Device::Type::kNvidia, 9>
    : public FusedQkNormRope {
 public:
  using FusedQkNormRope::FusedQkNormRope;
  using FusedQkNormRope::operator();

  static bool supports(Tensor qkv, int64_t, int64_t, int64_t, int64_t head_dim,
                       float, Tensor q_weight, Tensor k_weight,
                       Tensor cos_sin_cache, bool is_neox, Tensor position_ids,
                       int64_t forced_token_heads_per_warp) {
    return qkv.ndim() == 2 && qkv.IsContiguous() && head_dim == 128 &&
           cos_sin_cache.size(1) == head_dim &&
           qkv.dtype() == DataType::kBFloat16 &&
           q_weight.dtype() == qkv.dtype() && k_weight.dtype() == qkv.dtype() &&
           cos_sin_cache.dtype() == qkv.dtype() &&
           position_ids.dtype() == DataType::kInt64 && is_neox &&
           forced_token_heads_per_warp == -1;
  }

  std::size_t workspace_size_in_bytes() const override {
    return qkv_shape_[0] * qkv_shape_[1] * sizeof(std::uint16_t);
  }

  void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                  int64_t num_heads_v, int64_t head_dim, float eps,
                  Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache,
                  bool is_neox, Tensor position_ids,
                  int64_t /*forced_token_heads_per_warp*/) const override {
    const auto qk_heads = static_cast<std::uint64_t>(num_heads_q + num_heads_k);
    const auto total_heads =
        static_cast<std::uint64_t>(num_heads_q + num_heads_k + num_heads_v);
    const auto qkv_bytes = qkv.numel() * qkv.element_size();

    std::vector<std::uint64_t> qkv_sizes{
        static_cast<std::uint64_t>(qkv.size(0)), qk_heads,
        static_cast<std::uint64_t>(head_dim)};
    std::vector<std::int64_t> qkv_strides{
        static_cast<std::int64_t>(total_heads * head_dim), head_dim, 1};
    std::vector<std::uint64_t> weight_sizes{
        static_cast<std::uint64_t>(qkv.size(0)), qk_heads,
        static_cast<std::uint64_t>(head_dim)};
    std::vector<std::int64_t> q_weight_strides{0, 0, q_weight.stride(0)};
    std::vector<std::int64_t> k_weight_strides{0, 0, k_weight.stride(0)};
    std::vector<std::uint64_t> position_sizes{
        static_cast<std::uint64_t>(qkv.size(0)), qk_heads,
        static_cast<std::uint64_t>(head_dim)};
    std::vector<std::int64_t> position_strides{
        static_cast<std::int64_t>(position_ids.stride(0)), 0, 0};
    std::uint64_t scalar_shape[1] = {};
    std::int64_t scalar_strides[1] = {};

    void* output_workspace = workspace_;
    const bool owns_workspace = output_workspace == nullptr;
    if (owns_workspace) {
      const auto allocation_result = cudaMallocAsync(
          &output_workspace, qkv_bytes, static_cast<cudaStream_t>(stream_));
      if (allocation_result != cudaSuccess) {
        throw std::runtime_error(
            "NineToothed `FusedQkNormRope` workspace allocation failed");
      }
    }

    ninetoothed::Tensor qkv_input(const_cast<void*>(qkv.data()),
                                  qkv_sizes.data(), qkv_strides.data());
    ninetoothed::Tensor qkv_output(output_workspace, qkv_sizes.data(),
                                   qkv_strides.data());
    ninetoothed::Tensor q_weight_tensor(const_cast<void*>(q_weight.data()),
                                        weight_sizes.data(),
                                        q_weight_strides.data());
    ninetoothed::Tensor k_weight_tensor(const_cast<void*>(k_weight.data()),
                                        weight_sizes.data(),
                                        k_weight_strides.data());
    ninetoothed::Tensor position_tensor(const_cast<void*>(position_ids.data()),
                                        position_sizes.data(),
                                        position_strides.data());
    ninetoothed::Tensor cos_sin_cache_tensor(cos_sin_cache);
    double eps_value = static_cast<double>(eps);
    std::int64_t rotary_dim_value =
        static_cast<std::int64_t>(cos_sin_cache.size(1));
    std::int64_t is_neox_value = is_neox ? 1 : 0;
    ninetoothed::Tensor num_heads_q_tensor(num_heads_q, scalar_shape,
                                           scalar_strides);
    ninetoothed::Tensor eps_tensor(eps_value, scalar_shape, scalar_strides);
    ninetoothed::Tensor rotary_dim_tensor(rotary_dim_value, scalar_shape,
                                          scalar_strides);
    ninetoothed::Tensor is_neox_tensor(is_neox_value, scalar_shape,
                                       scalar_strides);

    const auto preload_result = cudaMemcpyAsync(
        output_workspace, qkv.data(), qkv_bytes, cudaMemcpyDeviceToDevice,
        static_cast<cudaStream_t>(stream_));
    if (preload_result != cudaSuccess) {
      if (owns_workspace) {
        cudaFreeAsync(output_workspace, static_cast<cudaStream_t>(stream_));
      }
      throw std::runtime_error(
          "NineToothed `FusedQkNormRope` workspace preload failed");
    }

    const auto launch_result = launch_infini_ops_ninetoothed_fused_qk_norm_rope(
        static_cast<NineToothedStream>(stream_), qkv_input, qkv_output,
        q_weight_tensor, k_weight_tensor, position_tensor, cos_sin_cache_tensor,
        num_heads_q_tensor, eps_tensor, rotary_dim_tensor, is_neox_tensor);
    if (launch_result != 0) {
      if (owns_workspace) {
        cudaFreeAsync(output_workspace, static_cast<cudaStream_t>(stream_));
      }
      throw std::runtime_error("NineToothed `FusedQkNormRope` launch failed");
    }

    const auto copy_result = cudaMemcpyAsync(
        qkv.data(), output_workspace, qkv_bytes, cudaMemcpyDeviceToDevice,
        static_cast<cudaStream_t>(stream_));
    if (owns_workspace) {
      cudaFreeAsync(output_workspace, static_cast<cudaStream_t>(stream_));
    }
    if (copy_result != cudaSuccess) {
      throw std::runtime_error(
          "NineToothed `FusedQkNormRope` output copy failed");
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_ROPE_H_
