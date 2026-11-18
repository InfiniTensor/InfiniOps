#ifndef INFINI_OPS_KUNLUN_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
#define INFINI_OPS_KUNLUN_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_

#include <stdexcept>

#include "base/fused_gated_delta_net_gating.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kKunlun>
    : public FusedGatedDeltaNetGating {
 public:
  using FusedGatedDeltaNetGating::FusedGatedDeltaNetGating;

  [[noreturn]] void operator()(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
                               float beta, float threshold, Tensor g,
                               Tensor beta_output) const override {
    throw std::runtime_error(
        "FusedGatedDeltaNetGating is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
