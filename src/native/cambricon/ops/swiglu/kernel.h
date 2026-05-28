#ifndef INFINI_OPS_CAMBRICON_SWIGLU_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SWIGLU_KERNEL_H_

#ifdef WITH_TORCH

#include "base/swiglu.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kCambricon> : public Swiglu {
 public:
  using Swiglu::Swiglu;

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    auto at_input = cambricon_torch_fallback::ToAten(input);
    auto at_gate = cambricon_torch_fallback::ToAten(gate);
    auto at_out = cambricon_torch_fallback::ToAten(out);

    auto run = [&](at::Tensor lhs, at::Tensor rhs) {
      auto work_lhs = lhs.to(at::kFloat);
      auto work_rhs = rhs.to(at::kFloat);
      auto result = work_lhs * (work_rhs * at::sigmoid(work_rhs));
      cambricon_torch_fallback::CopyToOutput(at_out, std::move(result));
    };

    try {
      run(at_input, at_gate);
    } catch (const c10::Error&) {
      run(at_input.cpu(), at_gate.cpu());
    }
  }
};

}  // namespace infini::ops

#endif

#endif
