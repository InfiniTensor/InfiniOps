#ifndef INFINI_OPS_CAMBRICON_CAST_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CAST_KERNEL_H_

#ifdef WITH_TORCH

#include "base/cast.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kCambricon> : public Cast {
 public:
  using Cast::Cast;

  void operator()(const Tensor input, Tensor out) const override {
    auto at_input = cambricon_torch_fallback::ToAten(input);
    auto at_out = cambricon_torch_fallback::ToAten(out);

    try {
      at_out.copy_(at_input.to(at_out.scalar_type()));
    } catch (const c10::Error&) {
      cambricon_torch_fallback::CopyToOutput(
          at_out, at_input.cpu().to(at_out.scalar_type()));
    }
  }
};

}  // namespace infini::ops

#endif

#endif
