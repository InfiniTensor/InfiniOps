#ifndef INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_H_
#define INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "base/fused_qk_norm.h"
#include "cuda_runtime_api.h"
#include "data_type.h"
#include "fused_qk_norm/infini_ops_ninetoothed_fused_qk_norm.h"
#include "ninetoothed/tensor.h"

namespace infini::ops {

static __global__ void PackFusedQkWeights(std::uint16_t* packed_weight,
                                          const std::uint16_t* query_weight,
                                          const std::uint16_t* key_weight,
                                          int64_t num_heads_q,
                                          int64_t num_heads_k,
                                          int64_t head_dim) {
  const auto index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto element_count = (num_heads_q + num_heads_k) * head_dim;
  if (index >= element_count) {
    return;
  }

  const auto head = index / head_dim;
  const auto dimension = index % head_dim;
  packed_weight[index] =
      head < num_heads_q ? query_weight[dimension] : key_weight[dimension];
}

template <>
class Operator<FusedQkNorm, Device::Type::kNvidia, 9> : public FusedQkNorm {
 public:
  using FusedQkNorm::FusedQkNorm;
  using FusedQkNorm::operator();

  Operator(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
           int64_t num_heads_v, int64_t head_dim, float eps, Tensor q_weight,
           Tensor k_weight)
      : FusedQkNorm(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
                    q_weight, k_weight),
        device_index_{qkv.device().index()} {
    const auto device_result = cudaSetDevice(device_index_);
    const auto packed_weight_bytes =
        static_cast<std::size_t>(num_heads_q + num_heads_k) *
        static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    assert(device_result == cudaSuccess &&
           "NineToothed `FusedQkNorm` device selection failed");
    const auto allocation_result =
        cudaMalloc(&default_packed_weight_, packed_weight_bytes);
    assert(allocation_result == cudaSuccess &&
           "NineToothed `FusedQkNorm` weight allocation failed");
  }

  ~Operator() override {
    const auto device_result = cudaSetDevice(device_index_);
    assert(device_result == cudaSuccess &&
           "NineToothed `FusedQkNorm` device selection failed");
    cudaFree(default_packed_weight_);
  }

  static bool supports(Tensor qkv, int64_t, int64_t, int64_t, int64_t head_dim,
                       float, Tensor q_weight, Tensor k_weight) {
    return qkv.ndim() == 2 && qkv.IsContiguous() && head_dim == 128 &&
           qkv.dtype() == DataType::kBFloat16 &&
           q_weight.dtype() == qkv.dtype() && k_weight.dtype() == qkv.dtype();
  }

  void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                  int64_t num_heads_v, int64_t head_dim, float eps,
                  Tensor q_weight, Tensor k_weight) const override {
    const auto qk_heads = static_cast<std::uint64_t>(num_heads_q + num_heads_k);
    const auto total_heads =
        static_cast<std::uint64_t>(num_heads_q + num_heads_k + num_heads_v);

    std::vector<std::uint64_t> qkv_sizes{
        static_cast<std::uint64_t>(qkv.size(0)), qk_heads,
        static_cast<std::uint64_t>(head_dim)};
    std::vector<std::int64_t> qkv_strides{
        static_cast<std::int64_t>(total_heads * head_dim), head_dim, 1};
    std::vector<std::uint64_t> weight_sizes{
        static_cast<std::uint64_t>(qkv.size(0)), qk_heads,
        static_cast<std::uint64_t>(head_dim)};
    std::vector<std::int64_t> weight_strides{
        0, static_cast<std::int64_t>(head_dim), 1};
    std::uint64_t scalar_shape[1] = {};
    std::int64_t scalar_strides[1] = {};
    double eps_value = static_cast<double>(eps);
    std::int64_t width_value = static_cast<std::int64_t>(head_dim);

    void* packed_weight = default_packed_weight_;

    const auto packed_weight_elements =
        static_cast<std::size_t>((num_heads_q + num_heads_k) * head_dim);
    const auto block_size = 128;
    const auto block_count =
        (packed_weight_elements + block_size - 1) / block_size;
    PackFusedQkWeights<<<static_cast<unsigned int>(block_count), block_size, 0,
                         static_cast<cudaStream_t>(stream_)>>>(
        static_cast<std::uint16_t*>(packed_weight),
        static_cast<const std::uint16_t*>(q_weight.data()),
        static_cast<const std::uint16_t*>(k_weight.data()), num_heads_q,
        num_heads_k, head_dim);
    assert(cudaGetLastError() == cudaSuccess &&
           "NineToothed `FusedQkNorm` weight packing launch failed");

    ninetoothed::Tensor qkv_input(const_cast<void*>(qkv.data()),
                                  qkv_sizes.data(), qkv_strides.data());
    ninetoothed::Tensor qkv_output(const_cast<void*>(qkv.data()),
                                   qkv_sizes.data(), qkv_strides.data());
    ninetoothed::Tensor weight_tensor(packed_weight, weight_sizes.data(),
                                      weight_strides.data());
    ninetoothed::Tensor eps_tensor(eps_value, scalar_shape, scalar_strides);
    ninetoothed::Tensor width_tensor(width_value, scalar_shape, scalar_strides);

    const auto result = launch_infini_ops_ninetoothed_fused_qk_norm(
        static_cast<NineToothedStream>(stream_), qkv_input, weight_tensor,
        eps_tensor, qkv_output, width_tensor);
    assert(result == 0 && "NineToothed `FusedQkNorm` launch failed");
  }

 private:
  void* default_packed_weight_{nullptr};

  int device_index_{-1};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_H_
