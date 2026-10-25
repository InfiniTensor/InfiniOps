#pragma once
#include "base/deepseek_moe.h"
namespace infini::ops {
template <>
class Operator<DeepseekMoe, Device::Type::kHygon, 8> : public DeepseekMoe {
 public:
  using DeepseekMoe::DeepseekMoe;
  void operator()(Tensor output, const Tensor hidden, const Tensor ids,
                  const Tensor weights, const std::vector<Tensor> gate,
                  const std::vector<Tensor> up, const std::vector<Tensor> down,
                  size_t intermediate, size_t num_experts) const override;
};
}  // namespace infini::ops
