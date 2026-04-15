#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_REGISTRY_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_REGISTRY_H_

#include "base/rotary_embedding.h"

namespace infini::ops {

// Implementation 0: `aclnnApplyRotaryPosEmbV2` (CANN, 2× IndexSelect + V2).
// Implementation 1: ATB `Rope` (fused kernel, eliminates GatherV3+Slice).
template <>
struct ActiveImplementationsImpl<RotaryEmbedding, Device::Type::kAscend> {
#if defined(INFINI_HAS_ATB)
  using type = List<0, 1>;
#else
  using type = List<0>;
#endif
};

}  // namespace infini::ops

#endif
