#ifndef INFINI_OPS_QY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_QY_EMBEDDING_KERNEL_H_

#include "native/cuda/ops/embedding/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kQy>
    : public CudaEmbedding<Runtime<Device::Type::kQy>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kQy>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
