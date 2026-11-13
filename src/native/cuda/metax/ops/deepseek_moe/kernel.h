#ifndef INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
#define INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_

#include <stdexcept>
#include <vector>

#include "base/deepseek_moe.h"

namespace infini::ops {

// DeepSeek MoE has no MetaX implementation yet. The operator
// is registered so that models referencing it link; calling
// it on MetaX throws.
template <>
class Operator<DeepseekMoe, Device::Type::kMetax, 0> : public DeepseekMoe {
 public:
  using DeepseekMoe::DeepseekMoe;
  using DeepseekMoe::operator();

  bool capture_safe() const override { return false; }

  void operator()(Tensor, const Tensor, const Tensor, const Tensor,
                  const std::vector<Tensor>, const std::vector<Tensor>,
                  const std::vector<Tensor>, size_t, size_t) const override {
    throw std::runtime_error("DeepseekMoe is not supported on MetaX");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
