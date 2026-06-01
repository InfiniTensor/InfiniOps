#ifndef INFINI_OPS_NVIDIA_MATMUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MATMUL_KERNEL_H_

#ifdef WITH_TORCH

#include "base/matmul.h"
#include "native/cuda/nvidia/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kNvidia> : public Matmul {
 public:
  using Matmul::Matmul;

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    auto at_a = nvidia_torch_fallback::ToAten(a);
    auto at_b = nvidia_torch_fallback::ToAten(b);
    auto at_c = nvidia_torch_fallback::ToAten(c);

    auto run = [&](at::Tensor lhs, at::Tensor rhs) {
      if (trans_a) {
        lhs = lhs.transpose(-2, -1);
      }

      if (trans_b) {
        rhs = rhs.transpose(-2, -1);
      }

      auto result = at::matmul(lhs.to(at::kFloat), rhs.to(at::kFloat));
      nvidia_torch_fallback::CopyToOutput(at_c, std::move(result));
    };

    try {
      run(at_a, at_b);
    } catch (const c10::Error&) {
      run(at_a.cpu(), at_b.cpu());
    }
  }
};

}  // namespace infini::ops

#endif

#endif
