#ifndef INFINI_OPS_KUNLUN_PER_TENSOR_QUANT_I8_KERNEL_H_
#define INFINI_OPS_KUNLUN_PER_TENSOR_QUANT_I8_KERNEL_H_

#include <stdexcept>

#include "base/per_tensor_quant_i8.h"

namespace infini::ops {

template <>
class Operator<PerTensorQuantI8, Device::Type::kKunlun>
    : public PerTensorQuantI8 {
 public:
  using PerTensorQuantI8::PerTensorQuantI8;

  [[noreturn]] void operator()(const Tensor x, Tensor packed, Tensor scale,
                               const std::optional<Tensor> zero,
                               const bool is_static) const override {
    throw std::runtime_error("PerTensorQuantI8 is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
