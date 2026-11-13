#ifndef INFINI_OPS_METAX_PER_CHANNEL_QUANT_I8_KERNEL_H_
#define INFINI_OPS_METAX_PER_CHANNEL_QUANT_I8_KERNEL_H_

#include <stdexcept>

#include "base/per_channel_quant_i8.h"

namespace infini::ops {

// Per-channel INT8 quantization has no MetaX implementation.
// The operator is registered so that models referencing it link;
// calling it on MetaX throws.
template <>
class Operator<PerChannelQuantI8, Device::Type::kMetax, 0>
    : public PerChannelQuantI8 {
 public:
  using PerChannelQuantI8::PerChannelQuantI8;
  using PerChannelQuantI8::operator();

  bool capture_safe() const override { return false; }

  void operator()(const Tensor x, Tensor packed, Tensor scale) const override {
    (void)x;
    (void)packed;
    (void)scale;
    throw std::runtime_error("PerChannelQuantI8 is not supported on MetaX");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_PER_CHANNEL_QUANT_I8_KERNEL_H_
