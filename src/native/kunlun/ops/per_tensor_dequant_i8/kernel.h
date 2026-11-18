#ifndef INFINI_OPS_KUNLUN_PER_TENSOR_DEQUANT_I8_KERNEL_H_
#define INFINI_OPS_KUNLUN_PER_TENSOR_DEQUANT_I8_KERNEL_H_

#include <stdexcept>

#include "base/per_tensor_dequant_i8.h"

namespace infini::ops {

template <>
class Operator<PerTensorDequantI8, Device::Type::kKunlun>
    : public PerTensorDequantI8 {
 public:
  using PerTensorDequantI8::PerTensorDequantI8;

  [[noreturn]] void operator()(
      Tensor x, const Tensor packed, const Tensor scale,
      const std::optional<Tensor> zero) const override {
    throw std::runtime_error("PerTensorDequantI8 is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
