#include <vector>

#include "native/cuda/metax/ops/cat/kernel.h"
#include "native/cuda/metax/ops/torch_fallback.h"

namespace infini::ops {

void Operator<Cat, Device::Type::kMetax>::operator()(
    const Tensor first_input, std::vector<Tensor> rest_inputs, int64_t /*dim*/,
    Tensor out) const {
  auto at_out = metax_torch_fallback::ToAten(out);
  std::vector<at::Tensor> at_inputs;
  at_inputs.reserve(1 + rest_inputs.size());
  at_inputs.push_back(metax_torch_fallback::ToAten(first_input));

  for (const auto& tensor : rest_inputs) {
    at_inputs.push_back(metax_torch_fallback::ToAten(tensor));
  }

  try {
    metax_torch_fallback::CopyToOutput(at_out, at::cat(at_inputs, dim_));
  } catch (const c10::Error&) {
    std::vector<at::Tensor> cpu_inputs;
    cpu_inputs.reserve(at_inputs.size());

    for (auto& tensor : at_inputs) {
      cpu_inputs.push_back(tensor.cpu());
    }

    metax_torch_fallback::CopyToOutput(at_out, at::cat(cpu_inputs, dim_));
  }
}

}  // namespace infini::ops
