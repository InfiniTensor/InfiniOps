#ifndef INFINI_OPS_KUNLUN_MOE_FUSED_DENSE_KERNEL_H_
#define INFINI_OPS_KUNLUN_MOE_FUSED_DENSE_KERNEL_H_

#include <stdexcept>

#include "base/moe_fused_dense.h"

namespace infini::ops {

// Placeholder: `MoeFusedDense` has no Kunlun implementation yet. The class
// must exist so downstream consumers (InfiniLM) that reference the operator
// type at compile time can build; running it throws.
template <>
class Operator<MoeFusedDense, Device::Type::kKunlun> : public MoeFusedDense {
 public:
  using MoeFusedDense::MoeFusedDense;

  void operator()(Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, size_t, const std::optional<Tensor>,
                  const std::optional<Tensor>) const override {
    throw std::runtime_error("MoeFusedDense is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
