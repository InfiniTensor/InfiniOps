#ifndef INFINI_OPS_METAX_MUL_KERNEL_H_
#define INFINI_OPS_METAX_MUL_KERNEL_H_

#ifdef WITH_TORCH

#include "base/mul.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMetax> : public Mul {
 public:
  using Mul::Mul;

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override;
};

}  // namespace infini::ops

#endif

#endif
