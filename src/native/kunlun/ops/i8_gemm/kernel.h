#ifndef INFINI_OPS_KUNLUN_I8_GEMM_KERNEL_H_
#define INFINI_OPS_KUNLUN_I8_GEMM_KERNEL_H_

#include <stdexcept>

#include "base/i8_gemm.h"

namespace infini::ops {

template <>
class Operator<I8Gemm, Device::Type::kKunlun> : public I8Gemm {
 public:
  using I8Gemm::I8Gemm;

  [[noreturn]] void operator()(
      Tensor c, const Tensor a, const Tensor as, const Tensor b,
      const Tensor bs, const std::optional<Tensor> bias) const override {
    throw std::runtime_error("I8Gemm is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
