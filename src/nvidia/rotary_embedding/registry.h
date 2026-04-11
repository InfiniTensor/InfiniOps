#ifndef INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_REGISTRY_H_
#define INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_REGISTRY_H_

#include "base/rotary_embedding.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<RotaryEmbedding, Device::Type::kNvidia> {
  using type = List<Impl::kDefault>;
};

}  // namespace infini::ops

#endif
