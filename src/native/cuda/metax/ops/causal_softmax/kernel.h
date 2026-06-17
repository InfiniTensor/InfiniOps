#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include <infini/rt/metax/runtime_.h>
#include "native/cuda/metax/runtime_utils.h"
#include "native/cuda/ops/causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kMetax>
    : public CudaCausalSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kMetax>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
