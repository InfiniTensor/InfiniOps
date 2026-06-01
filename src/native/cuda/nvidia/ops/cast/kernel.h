#ifndef INFINI_OPS_NVIDIA_CAST_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAST_KERNEL_H_

#ifdef WITH_TORCH

#include "base/cast.h"
#include "native/cuda/nvidia/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kNvidia> : public Cast {
 public:
  using Cast::Cast;

  void operator()(const Tensor input, Tensor out) const override {
    auto at_input = nvidia_torch_fallback::ToAten(input);
    auto at_out = nvidia_torch_fallback::ToAten(out);

    try {
      at_out.copy_(at_input.to(at_out.scalar_type()));
    } catch (const c10::Error&) {
      nvidia_torch_fallback::CopyToOutput(
          at_out, at_input.cpu().to(at_out.scalar_type()));
    }
  }
};

}  // namespace infini::ops

#endif

#endif
