#include "native/cuda/metax/ops/matmul/kernel.h"
#include "native/cuda/metax/ops/torch_fallback.h"

namespace infini::ops {

void Operator<Matmul, Device::Type::kMetax>::operator()(const Tensor a,
                                                        const Tensor b,
                                                        Tensor c, bool trans_a,
                                                        bool trans_b) const {
  auto at_a = metax_torch_fallback::ToAten(a);
  auto at_b = metax_torch_fallback::ToAten(b);
  auto at_c = metax_torch_fallback::ToAten(c);

  auto run = [&](at::Tensor lhs, at::Tensor rhs) {
    if (trans_a) {
      lhs = lhs.transpose(-2, -1);
    }

    if (trans_b) {
      rhs = rhs.transpose(-2, -1);
    }

    auto result = at::matmul(lhs.to(at::kFloat), rhs.to(at::kFloat));
    metax_torch_fallback::CopyToOutput(at_c, std::move(result));
  };

  try {
    run(at_a, at_b);
  } catch (const c10::Error&) {
    run(at_a.cpu(), at_b.cpu());
  }
}

}  // namespace infini::ops
