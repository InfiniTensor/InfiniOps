#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#ifdef WITH_TORCH

#include "base/mul.h"
#include "native/cuda/nvidia/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia> : public Mul {
 public:
  using Mul::Mul;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto at_input = nvidia_torch_fallback::ToAten(input);
    auto at_other = nvidia_torch_fallback::ToAten(other);
    auto at_out = nvidia_torch_fallback::ToAten(out);

    try {
      at::mul_out(at_out, at_input, at_other);
    } catch (const c10::Error&) {
      auto result = at::mul(at_input.cpu(), at_other.cpu());
      nvidia_torch_fallback::CopyToOutput(at_out, std::move(result));
    }
  }
};

}  // namespace infini::ops

#endif

#endif
