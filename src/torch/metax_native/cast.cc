#include "native/cuda/metax/ops/cast/kernel.h"

#include "native/cuda/metax/ops/torch_fallback.h"

namespace infini::ops {

void Operator<Cast, Device::Type::kMetax>::operator()(const Tensor input,
                                                       Tensor out) const {
  auto at_input = metax_torch_fallback::ToAten(input);
  auto at_out = metax_torch_fallback::ToAten(out);

  try {
    at_out.copy_(at_input.to(at_out.scalar_type()));
  } catch (const c10::Error&) {
    metax_torch_fallback::CopyToOutput(
        at_out, at_input.cpu().to(at_out.scalar_type()));
  }
}

}  // namespace infini::ops
