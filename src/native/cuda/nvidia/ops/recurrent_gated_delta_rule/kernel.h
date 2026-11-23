#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

#include "base/recurrent_gated_delta_rule.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/recurrent_gated_delta_rule/kernel.h"

namespace infini::ops {

// NVIDIA provider for the shared CUDA single-token decode gated-delta-rule
// kernel. The shared host path performs the shape/extent validation;
// NVIDIA adds the device guard, the per-device block-size precheck, and
// the post-launch error check.
template <>
class Operator<RecurrentGatedDeltaRule, Device::Type::kNvidia>
    : public CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kNvidia>> {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
           const Tensor beta, Tensor initial_state,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_state, bool use_qk_l2norm)
      : CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kNvidia>>(
            q, k, v, g, beta, initial_state, initial_state_indices,
            final_state_indices, out, final_state, use_qk_l2norm) {
    if (device_.type() != Device::Type::kNvidia)
      throw std::invalid_argument("recurrent delta requires NVIDIA");
    CheckDevice();
    cudaDeviceProp property{};
    if (cudaGetDeviceProperties(&property, device_.index()) != cudaSuccess ||
        property.maxThreadsPerBlock < 256)
      throw std::runtime_error(
          "recurrent delta requires 256-thread CUDA blocks");
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, Tensor initial_state,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_state,
                  bool use_qk_l2norm) const override {
    CheckDevice();
    CudaRecurrentGatedDeltaRule<Runtime<Device::Type::kNvidia>>::operator()(
        q, k, v, g, beta, initial_state, initial_state_indices,
        final_state_indices, out, final_state, use_qk_l2norm);
    if (cudaPeekAtLastError() != cudaSuccess)
      throw std::runtime_error("recurrent delta CUDA launch failed");
  }

 private:
  void CheckDevice() const {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("recurrent delta device query failed");
    if (device != device_.index())
      throw std::invalid_argument("recurrent delta owner device mismatch");
  }
};

}  // namespace infini::ops
