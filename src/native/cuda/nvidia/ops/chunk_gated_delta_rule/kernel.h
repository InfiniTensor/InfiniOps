#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

#include "base/chunk_gated_delta_rule.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/chunk_gated_delta_rule/kernel.h"

namespace infini::ops {

// NVIDIA provider for the shared CUDA chunked-prefill gated-delta-rule
// kernel. The shared host path performs the shape/extent/workspace
// validation; NVIDIA adds the device guard, the per-device block-size
// precheck, and the post-launch error check.
template <>
class Operator<ChunkGatedDeltaRule, Device::Type::kNvidia>
    : public CudaChunkGatedDeltaRule<Runtime<Device::Type::kNvidia>> {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
           const Tensor beta, Tensor initial_state,
           const std::optional<Tensor> cu_seqlens,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_state, bool use_qk_l2norm,
           size_t chunk_size)
      : CudaChunkGatedDeltaRule<Runtime<Device::Type::kNvidia>>(
            q, k, v, g, beta, initial_state, cu_seqlens, initial_state_indices,
            final_state_indices, out, final_state, use_qk_l2norm, chunk_size) {
    if (device_.type() != Device::Type::kNvidia)
      throw std::invalid_argument("chunk delta requires NVIDIA");
    CheckDevice();
    cudaDeviceProp property{};
    if (cudaGetDeviceProperties(&property, device_.index()) != cudaSuccess ||
        property.maxThreadsPerBlock < int(D_))
      throw std::runtime_error("chunk delta CUDA block size unsupported");
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, Tensor initial_state,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_state, bool use_qk_l2norm,
                  size_t chunk_size) const override {
    CheckDevice();
    CudaChunkGatedDeltaRule<Runtime<Device::Type::kNvidia>>::operator()(
        q, k, v, g, beta, initial_state, cu_seqlens, initial_state_indices,
        final_state_indices, out, final_state, use_qk_l2norm, chunk_size);
    if (cudaPeekAtLastError() != cudaSuccess)
      throw std::runtime_error("chunk delta CUDA launch failed");
  }

 private:
  void CheckDevice() const {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("chunk delta device query failed");
    if (device != device_.index())
      throw std::invalid_argument("chunk delta owner device mismatch");
  }
};

}  // namespace infini::ops
