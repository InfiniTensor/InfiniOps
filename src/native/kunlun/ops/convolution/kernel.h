#ifndef INFINI_OPS_KUNLUN_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_KUNLUN_CONVOLUTION_KERNEL_H_

#include <stdexcept>

#include "base/convolution.h"

namespace infini::ops {

// Placeholder:  has no Kunlun implementation yet. The class must
// exist so downstream consumers (InfiniLM) referencing the operator type
// at compile time build; running it throws.
template <>
class Operator<Convolution, Device::Type::kKunlun> : public Convolution {
 public:
  using Convolution::Convolution;

  [[noreturn]] void operator()(
      const Tensor input, const Tensor weight, std::optional<Tensor> bias,
      const std::vector<int64_t> stride, const std::vector<int64_t> padding,
      const std::vector<int64_t> dilation, const bool transposed,
      const std::vector<int64_t> output_padding, const int64_t groups,
      Tensor out) const override {
    throw std::runtime_error("Convolution is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
