#ifndef INFINI_OPS_HYGON_EMBEDDING_KERNEL_H_
#define INFINI_OPS_HYGON_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kHygon>
    : public CudaEmbedding<Runtime<Device::Type::kHygon>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kHygon>>::CudaEmbedding;
  bool capture_safe() const override { return !max_norm_.has_value(); }
};

}  // namespace infini::ops

#endif
