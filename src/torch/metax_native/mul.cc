#include "native/cuda/metax/ops/mul/kernel.h"
#include "native/cuda/metax/ops/torch_fallback.h"

namespace infini::ops {

void Operator<Mul, Device::Type::kMetax>::operator()(const Tensor input,
                                                     const Tensor other,
                                                     Tensor out) const {
  auto at_input = metax_torch_fallback::ToAten(input);
  auto at_other = metax_torch_fallback::ToAten(other);
  auto at_out = metax_torch_fallback::ToAten(out);

  try {
    at::mul_out(at_out, at_input, at_other);
  } catch (const c10::Error&) {
    auto result = at::mul(at_input.cpu(), at_other.cpu());
    metax_torch_fallback::CopyToOutput(at_out, std::move(result));
  }
}

}  // namespace infini::ops
