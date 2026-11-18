#ifndef INFINI_OPS_KUNLUN_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#define INFINI_OPS_KUNLUN_MAMBA_SELECTIVE_SCAN_KERNEL_H_

#include <stdexcept>

#include "base/mamba_selective_scan.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<MambaSelectiveScan, Device::Type::kKunlun>
    : public MambaSelectiveScan {
 public:
  using MambaSelectiveScan::MambaSelectiveScan;

  [[noreturn]] void operator()(Tensor out, const Tensor x, const Tensor dt,
                               const Tensor b, const Tensor c,
                               const Tensor a_log, const Tensor d,
                               const Tensor gate, const Tensor dt_bias,
                               Tensor state) const override {
    throw std::runtime_error("MambaSelectiveScan is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
