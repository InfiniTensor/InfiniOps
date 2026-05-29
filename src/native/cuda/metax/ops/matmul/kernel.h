#ifndef INFINI_OPS_METAX_MATMUL_KERNEL_H_
#define INFINI_OPS_METAX_MATMUL_KERNEL_H_

#ifdef WITH_TORCH

#include "base/matmul.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kMetax> : public Matmul {
 public:
  using Matmul::Matmul;

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override;
};

}  // namespace infini::ops

#endif

#endif
