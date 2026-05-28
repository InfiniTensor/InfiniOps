#ifndef INFINI_OPS_CAMBRICON_LINEAR_KERNEL_H_
#define INFINI_OPS_CAMBRICON_LINEAR_KERNEL_H_

#ifdef WITH_TORCH

#include "base/linear.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kCambricon> : public Linear {
 public:
  using Linear::Linear;

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    auto at_a = cambricon_torch_fallback::ToAten(a);
    auto at_b = cambricon_torch_fallback::ToAten(b);
    auto at_bias = cambricon_torch_fallback::ToAten(bias);
    auto at_out = cambricon_torch_fallback::ToAten(out);

    auto run = [&](at::Tensor lhs, at::Tensor rhs,
                   std::optional<at::Tensor> bias_tensor) {
      if (trans_a) {
        lhs = lhs.transpose(-2, -1);
      }

      if (trans_b) {
        rhs = rhs.transpose(-2, -1);
      }

      auto result = at::matmul(lhs.to(at::kFloat), rhs.to(at::kFloat));

      if (bias_tensor.has_value()) {
        result.add_(bias_tensor->to(result.device()).to(at::kFloat));
      }

      cambricon_torch_fallback::CopyToOutput(at_out, std::move(result));
    };

    try {
      run(at_a, at_b, at_bias);
    } catch (const c10::Error&) {
      std::optional<at::Tensor> cpu_bias;

      if (at_bias.has_value()) {
        cpu_bias = at_bias->cpu();
      }

      run(at_a.cpu(), at_b.cpu(), cpu_bias);
    }
  }
};

}  // namespace infini::ops

#endif

#endif
