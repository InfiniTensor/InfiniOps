#ifndef INFINI_OPS_KUNLUN_DEEPSEEK_MOE_KERNEL_H_
#define INFINI_OPS_KUNLUN_DEEPSEEK_MOE_KERNEL_H_

#include <stdexcept>

#include "base/deepseek_moe.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<DeepseekMoe, Device::Type::kKunlun> : public DeepseekMoe {
 public:
  using DeepseekMoe::DeepseekMoe;

  [[noreturn]] void operator()(Tensor output, const Tensor hidden,
                               const Tensor ids, const Tensor weights,
                               const std::vector<Tensor> gate,
                               const std::vector<Tensor> up,
                               const std::vector<Tensor> down,
                               size_t intermediate,
                               size_t num_experts) const override {
    throw std::runtime_error("DeepseekMoe is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
