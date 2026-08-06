#ifndef INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_
#define INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_

#include "base/silu_and_mul.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchSiluAndMul : public ::infini::ops::SiluAndMul {
 public:
  TorchSiluAndMul(const Tensor input, Tensor out);

  using ::infini::ops::SiluAndMul::operator();

  void operator()(const Tensor input, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_
