#ifndef INFINI_OPS_KUNLUN_MROPE_KERNEL_H_
#define INFINI_OPS_KUNLUN_MROPE_KERNEL_H_

#include <stdexcept>

#include "base/mrope.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<Mrope, Device::Type::kKunlun> : public Mrope {
 public:
  using Mrope::Mrope;

  [[noreturn]] void operator()(const Tensor q, const Tensor k, const Tensor cos,
                               const Tensor sin, const Tensor p, int head,
                               int rot, int st, int sh, int sw,
                               bool interleaved, Tensor qo,
                               Tensor ko) const override {
    throw std::runtime_error("Mrope is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
