#ifndef INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "base/moe_topk_softmax.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/moe_topk_softmax/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

// MoE top-k gating with softmax normalization. For power-of-two expert
// counts (up to 512) a fused, vectorized kernel performs softmax and top-k
// selection in a single pass; otherwise the generic softmax + top-k path is
// used and requires a workspace of num_tokens * num_experts floats.
template <typename Backend>
class CudaMoeTopkSoftmax : public MoeTopkSoftmax {
 public:
  CudaMoeTopkSoftmax(Tensor topk_weights, Tensor topk_indices,
                     Tensor gating_output, Tensor correction_bias,
                     bool renormalize, float moe_softcapping)
      : MoeTopkSoftmax(topk_weights, topk_indices, gating_output,
                       correction_bias, renormalize, moe_softcapping),
        gating_output_type_{gating_output.dtype()} {}

  std::size_t workspace_size_in_bytes() const override {
    if (MoeTopkSoftmaxNeedsWorkspace(num_experts_)) {
      return num_tokens_ * num_experts_ * sizeof(float);
    }
    return 0;
  }

  void operator()(Tensor topk_weights, Tensor topk_indices,
                  Tensor gating_output, Tensor correction_bias,
                  bool renormalize, float moe_softcapping) const override {
    if (num_tokens_ == 0) {
      return;
    }

    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    const float* correction_bias_ptr =
        has_correction_bias_
            ? reinterpret_cast<const float*>(correction_bias.data())
            : nullptr;

    DispatchFunc<Backend::kDeviceType, DataType::kFloat16, DataType::kBFloat16,
                 DataType::kFloat32>(
        gating_output_type_,
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;
          LaunchMoeTopkSoftmax<Backend::kDeviceType, T>(
              reinterpret_cast<const T*>(gating_output.data()),
              reinterpret_cast<float*>(topk_weights.data()),
              reinterpret_cast<int*>(topk_indices.data()),
              reinterpret_cast<float*>(workspace_),
              static_cast<int>(num_tokens_), static_cast<int>(num_experts_),
              static_cast<int>(topk_), renormalize, moe_softcapping,
              correction_bias_ptr, stream);
        },
        "CudaMoeTopkSoftmax::operator()");
  }

 private:
  DataType gating_output_type_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_TOPK_SOFTMAX_KERNEL_H_