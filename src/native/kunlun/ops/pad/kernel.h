#ifndef INFINI_OPS_KUNLUN_PAD_KERNEL_H_
#define INFINI_OPS_KUNLUN_PAD_KERNEL_H_

#include <stdexcept>

#include "base/pad.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<Pad, Device::Type::kKunlun> : public Pad {
 public:
  using Pad::Pad;

  [[noreturn]] void operator()(const Tensor input,
                               const std::vector<int64_t> pad,
                               const std::string mode,
                               const std::optional<double> value,
                               Tensor out) const override {
    throw std::runtime_error("Pad is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
