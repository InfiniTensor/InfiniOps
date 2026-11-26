#pragma once
#include "base/moe_fused_dense.h"
namespace infini::ops {
template <Device::Type kDev>
class Operator<MoeFusedDense, kDev, 2> : public MoeFusedDense {
 public:
  using MoeFusedDense::MoeFusedDense;
  using MoeFusedDense::operator();
  void operator()(Tensor output, const Tensor hidden, const Tensor w13,
                  const Tensor w2, const Tensor weights, const Tensor ids,
                  const Tensor sorted, const Tensor experts, const Tensor total,
                  size_t block_size, const std::optional<Tensor> bias1 = std::nullopt,
                  const std::optional<Tensor> bias2 = std::nullopt) const override;
};
}  // namespace infini::ops
