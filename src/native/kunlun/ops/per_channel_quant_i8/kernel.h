#ifndef INFINI_OPS_KUNLUN_PER_CHANNEL_QUANT_I8_KERNEL_H_
#define INFINI_OPS_KUNLUN_PER_CHANNEL_QUANT_I8_KERNEL_H_

#include <stdexcept>

#include "base/per_channel_quant_i8.h"

namespace infini::ops {

template <>
class Operator<PerChannelQuantI8, Device::Type::kKunlun>
    : public PerChannelQuantI8 {
 public:
  using PerChannelQuantI8::PerChannelQuantI8;

  [[noreturn]] void operator()(const Tensor x, Tensor packed,
                               Tensor scale) const override {
    throw std::runtime_error("PerChannelQuantI8 is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
