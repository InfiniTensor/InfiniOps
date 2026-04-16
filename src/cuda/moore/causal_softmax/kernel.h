#ifndef INFINI_OPS_MOORE_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MOORE_CAUSAL_SOFTMAX_KERNEL_H_

// clang-format off
#include <musa_runtime.h>
// clang-format on

// clang-format off
#include "moore/device_.h"
// clang-format on

#include "cuda/causal_softmax/kernel.h"
#include "moore/caster.cuh"
#include "moore/runtime_.h"

namespace infini::ops {

struct MooreCausalSoftmaxBackend : Runtime<Device::Type::kMoore> {
  // Moore's causal softmax kernel should not dispatch block sizes above 1024.
  static constexpr int max_block_size = 1024;
};

template <>
class Operator<CausalSoftmax, Device::Type::kMoore>
    : public CudaCausalSoftmax<MooreCausalSoftmaxBackend> {
 public:
  using CudaCausalSoftmax<MooreCausalSoftmaxBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
