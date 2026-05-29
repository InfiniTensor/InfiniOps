#ifndef INFINI_OPS_METAX_LINEAR_KERNEL_H_
#define INFINI_OPS_METAX_LINEAR_KERNEL_H_

#ifdef WITH_TORCH

#include "base/linear.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kMetax> : public Linear {
 public:
  using Linear::Linear;

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override;
};

}  // namespace infini::ops

#endif

#endif
