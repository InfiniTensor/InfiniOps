#ifndef INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_REGISTRY_H_
#define INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_REGISTRY_H_

#include "base/apply_rotary_pos_emb.h"

namespace infini::ops {

// Implementation 0: `aclnnApplyRotaryPosEmbV2` (CANN, apply-only).
// Implementation 1: ATB `Rope` (fused kernel, apply-only).
template <>
struct ActiveImplementationsImpl<ApplyRotaryPosEmb, Device::Type::kAscend> {
#if defined(INFINI_HAS_ATB)
  using type = List<0, 1>;
#else
  using type = List<0>;
#endif
};

}  // namespace infini::ops

#endif
