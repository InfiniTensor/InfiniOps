#ifndef INFINI_OPS_ASCEND_PAGED_ATTENTION_REGISTRY_H_
#define INFINI_OPS_ASCEND_PAGED_ATTENTION_REGISTRY_H_

#include "base/paged_attention.h"

namespace infini::ops {

// ATB `PagedAttentionParam` is the only implementation.  Unlike
// `FlashAttention`, paged attention exists specifically to provide a
// graph-safe decode path (all parameters are tensor-based, no
// `aclIntArray*`).  When ATB is unavailable, fall back to
// `FlashAttention` for decode at the Python layer.
template <>
struct ActiveImplementationsImpl<PagedAttention, Device::Type::kAscend> {
#ifdef INFINI_HAS_ATB
  using type = List<0>;
#else
  using type = List<>;
#endif
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_PAGED_ATTENTION_REGISTRY_H_
