#ifndef INFINI_OPS_METAX_INTERNAL_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_METAX_INTERNAL_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/internal_causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<internal::CausalSoftmax, Device::Type::kMetax>
    : public internal::CudaCausalSoftmax<Runtime<Device::Type::kMetax>> {
 public:
  using internal::CudaCausalSoftmax<
      Runtime<Device::Type::kMetax>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
