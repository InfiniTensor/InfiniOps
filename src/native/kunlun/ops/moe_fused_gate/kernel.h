#ifndef INFINI_OPS_KUNLUN_MOE_FUSED_GATE_KERNEL_H_
#define INFINI_OPS_KUNLUN_MOE_FUSED_GATE_KERNEL_H_

#include <stdexcept>

#include "base/moe_fused_gate.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<MoeFusedGate, Device::Type::kKunlun> : public MoeFusedGate {
 public:
  using MoeFusedGate::MoeFusedGate;

  [[noreturn]] void operator()(Tensor weights, Tensor indices,
                               const Tensor input, const Tensor bias,
                               size_t groups, size_t selected_groups,
                               size_t shared, float scale,
                               bool apply_scale) const override {
    throw std::runtime_error("MoeFusedGate is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
