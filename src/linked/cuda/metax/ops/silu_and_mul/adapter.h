#ifndef INFINI_OPS_LINKED_CUDA_METAX_OPS_SILU_AND_MUL_ADAPTER_H_
#define INFINI_OPS_LINKED_CUDA_METAX_OPS_SILU_AND_MUL_ADAPTER_H_

#include "base/silu_and_mul.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMetax, 11> : public SiluAndMul {
 public:
  Operator(const Tensor input, Tensor out);

  using SiluAndMul::operator();

  void operator()(const Tensor input, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_CUDA_METAX_OPS_SILU_AND_MUL_ADAPTER_H_
