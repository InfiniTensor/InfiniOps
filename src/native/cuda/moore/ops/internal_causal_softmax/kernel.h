#ifndef INFINI_OPS_MOORE_INTERNAL_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MOORE_INTERNAL_CAUSAL_SOFTMAX_KERNEL_H_

// clang-format off
#include <musa_runtime.h>
// clang-format on

// clang-format off
#include "native/cuda/moore/device_.h"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/internal_causal_softmax/kernel.h"

namespace infini::ops::internal {

struct MooreCausalSoftmaxBackend : Runtime<Device::Type::kMoore> {
  // Moore's causal softmax kernel should not dispatch block sizes above 1024.
  static constexpr int max_block_size = 1024;
};

}  // namespace infini::ops::internal

namespace infini::ops {

template <>
class Operator<internal::CausalSoftmax, Device::Type::kMoore>
    : public internal::CudaCausalSoftmax<internal::MooreCausalSoftmaxBackend> {
 public:
  using internal::CudaCausalSoftmax<
      internal::MooreCausalSoftmaxBackend>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
