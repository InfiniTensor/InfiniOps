#ifndef INFINI_OPS_METAX_I8_GEMM_KERNEL_H_
#define INFINI_OPS_METAX_I8_GEMM_KERNEL_H_

#include <stdexcept>

#include "base/i8_gemm.h"

namespace infini::ops {

// INT8 GEMM has no MetaX implementation. The operator is
// registered so that models referencing it link; calling it on
// MetaX throws.
template <>
class Operator<I8Gemm, Device::Type::kMetax, 0> : public I8Gemm {
 public:
  using I8Gemm::I8Gemm;
  using I8Gemm::operator();

  bool capture_safe() const override { return false; }

  void operator()(Tensor c, const Tensor a, const Tensor as, const Tensor b,
                  const Tensor bs,
                  const std::optional<Tensor> bias) const override {
    (void)c;
    (void)a;
    (void)as;
    (void)b;
    (void)bs;
    (void)bias;
    throw std::runtime_error("I8Gemm is not supported on MetaX");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_I8_GEMM_KERNEL_H_
