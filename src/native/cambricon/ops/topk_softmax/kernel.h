#ifndef INFINI_OPS_CAMBRICON_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CAMBRICON_TOPK_SOFTMAX_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "base/topk_softmax.h"
#include "dispatcher.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T, typename Index>
void TopkSoftmaxUnion(cnrtQueue_t queue, int core_per_cluster,
                      int cluster_count, const T* gating_output,
                      const float* bias, const uint8_t* is_padding,
                      float* topk_weights, Index* topk_indices,
                      int32_t* token_expert_indices, int32_t num_tokens,
                      int32_t num_experts, int32_t topk, bool renormalize);

template <>
class Operator<TopkSoftmax, Device::Type::kCambricon> : public TopkSoftmax {
 public:
  Operator(const Tensor gating_output, std::optional<Tensor> bias,
           std::optional<Tensor> is_padding, const bool renormalize,
           Tensor topk_weights, Tensor topk_indices,
           Tensor token_expert_indices)
      : TopkSoftmax{gating_output,       bias,         is_padding,
                    renormalize,         topk_weights, topk_indices,
                    token_expert_indices} {
    cnrt_utils::GetLaunchConfig(gating_output.device(), &core_per_cluster_,
                                &cluster_count_);
  }

  void operator()(const Tensor gating_output, std::optional<Tensor> bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  Tensor topk_weights, Tensor topk_indices,
                  Tensor token_expert_indices) const override {
    ValidateCallMetadata(gating_output, bias, is_padding, renormalize,
                         topk_weights, topk_indices, token_expert_indices);
    if (num_tokens_ == 0) {
      return;
    }

    const auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    using InputTypes =
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>;
    using IndexTypes =
        List<DataType::kInt32, DataType::kUInt32, DataType::kInt64>;
    DispatchFunc<Device::Type::kCambricon, InputTypes, IndexTypes>(
        {input_dtype_, index_dtype_},
        [&](auto input_tag, auto index_tag) {
          using T = typename decltype(input_tag)::type;
          using Index = typename decltype(index_tag)::type;
          TopkSoftmaxUnion<T, Index>(
              queue, core_per_cluster_, cluster_count_,
              static_cast<const T*>(gating_output.data()),
              bias ? static_cast<const float*>(bias->data()) : nullptr,
              is_padding ? static_cast<const uint8_t*>(is_padding->data())
                         : nullptr,
              static_cast<float*>(topk_weights.data()),
              static_cast<Index*>(topk_indices.data()),
              static_cast<int32_t*>(token_expert_indices.data()),
              static_cast<int32_t>(num_tokens_),
              static_cast<int32_t>(num_experts_), static_cast<int32_t>(topk_),
              renormalize_);
        },
        "CambriconTopkSoftmax::operator()");
  }

 private:
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_TOPK_SOFTMAX_KERNEL_H_
