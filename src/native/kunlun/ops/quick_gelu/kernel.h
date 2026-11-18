#ifndef INFINI_OPS_KUNLUN_QUICK_GELU_KERNEL_H_
#define INFINI_OPS_KUNLUN_QUICK_GELU_KERNEL_H_

#include <stdexcept>

#include "base/quick_gelu.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<QuickGelu, Device::Type::kKunlun> : public QuickGelu {
 public:
  using QuickGelu::QuickGelu;

  [[noreturn]] void operator()(const Tensor input, Tensor out) const override {
    throw std::runtime_error("QuickGelu is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
