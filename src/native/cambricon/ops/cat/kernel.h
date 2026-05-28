#ifndef INFINI_OPS_CAMBRICON_CAT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CAT_KERNEL_H_

#ifdef WITH_TORCH

#include <vector>

#include "base/cat.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kCambricon> : public Cat {
 public:
  Operator(const Tensor first_input, std::vector<Tensor> rest_inputs,
           int64_t dim, Tensor out)
      : Cat{first_input, std::move(rest_inputs), dim, out} {}

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t /*dim*/, Tensor out) const override {
    auto at_out = cambricon_torch_fallback::ToAten(out);
    std::vector<at::Tensor> at_inputs;
    at_inputs.reserve(1 + rest_inputs.size());
    at_inputs.push_back(cambricon_torch_fallback::ToAten(first_input));

    for (const auto& tensor : rest_inputs) {
      at_inputs.push_back(cambricon_torch_fallback::ToAten(tensor));
    }

    try {
      cambricon_torch_fallback::CopyToOutput(at_out, at::cat(at_inputs, dim_));
    } catch (const c10::Error&) {
      std::vector<at::Tensor> cpu_inputs;
      cpu_inputs.reserve(at_inputs.size());

      for (auto& tensor : at_inputs) {
        cpu_inputs.push_back(tensor.cpu());
      }

      cambricon_torch_fallback::CopyToOutput(at_out, at::cat(cpu_inputs, dim_));
    }
  }
};

}  // namespace infini::ops

#endif

#endif
