#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "cuda/causal_softmax/kernel.h"
#include "cuda/metax/caster.cuh"
#include "cuda/metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kMetax>
    : public CudaCausalSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kMetax>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
