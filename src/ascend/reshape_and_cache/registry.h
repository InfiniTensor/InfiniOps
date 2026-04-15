#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_REGISTRY_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_REGISTRY_H_

#include "base/reshape_and_cache.h"

namespace infini::ops {

// Implementation 0: `aclnnInplaceIndexCopy` (CANN 8.0+, two calls for K+V).
// Implementation 1: `aclnnScatterPaKvCache` (CANN 8.5.1+, single fused call).
// Implementation 2: ATB `ReshapeAndCacheNdKernel` (fused K+V, graph-safe).
template <>
struct ActiveImplementationsImpl<ReshapeAndCache, Device::Type::kAscend> {
#if defined(INFINI_HAS_ATB) && \
    __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")
  using type = List<0, 1, 2>;
#elif defined(INFINI_HAS_ATB)
  using type = List<0, 2>;
#elif __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")
  using type = List<0, 1>;
#else
  using type = List<0>;
#endif
};

}  // namespace infini::ops

#endif
