#ifndef INFINI_OPS_KUNLUN_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_KUNLUN_LAYER_NORM_KERNEL_H_

#include <stdexcept>

#include "base/layer_norm.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<LayerNorm, Device::Type::kKunlun> : public LayerNorm {
 public:
  using LayerNorm::LayerNorm;

  [[noreturn]] void operator()(const Tensor input, const Tensor weight,
                               const std::optional<Tensor> bias, float eps,
                               Tensor out) const override {
    throw std::runtime_error("LayerNorm is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
