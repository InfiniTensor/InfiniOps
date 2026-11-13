#ifndef INFINI_OPS_METAX_MOE_FUSED_GATE_KERNEL_H_
#define INFINI_OPS_METAX_MOE_FUSED_GATE_KERNEL_H_

#include <stdexcept>

#include "base/moe_fused_gate.h"

namespace infini::ops {

// MoE fused gate (DeepSeek-style grouped top-k) has no MetaX
// implementation yet. The operator is registered so that models
// referencing it link; calling it on MetaX throws.
template <>
class Operator<MoeFusedGate, Device::Type::kMetax, 0> : public MoeFusedGate {
 public:
  using MoeFusedGate::MoeFusedGate;
  using MoeFusedGate::operator();

  bool capture_safe() const override { return false; }

  void operator()(Tensor, Tensor, const Tensor, const Tensor, size_t, size_t,
                  size_t, float, bool) const override {
    throw std::runtime_error("MoeFusedGate is not supported on MetaX");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_MOE_FUSED_GATE_KERNEL_H_
