#ifndef INFINI_OPS_CAMBRICON_MUL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MUL_KERNEL_H_

#ifdef WITH_TORCH

#include "base/mul.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kCambricon> : public Mul {
 public:
  using Mul::Mul;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto at_input = cambricon_torch_fallback::ToAten(input);
    auto at_other = cambricon_torch_fallback::ToAten(other);
    auto at_out = cambricon_torch_fallback::ToAten(out);

    try {
      at::mul_out(at_out, at_input, at_other);
    } catch (const c10::Error&) {
      auto result = at::mul(at_input.cpu(), at_other.cpu());
      cambricon_torch_fallback::CopyToOutput(at_out, std::move(result));
    }
  }
};

}  // namespace infini::ops

#endif

#endif
