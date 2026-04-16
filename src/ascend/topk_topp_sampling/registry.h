#ifndef INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_REGISTRY_H_
#define INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_REGISTRY_H_

#include "base/topk_topp_sampling.h"

namespace infini::ops {

// Implementation 0: ATB `TopkToppSamplingParam`
// (BATCH_TOPK_EXPONENTIAL_SAMPLING).
template <>
struct ActiveImplementationsImpl<TopkToppSampling, Device::Type::kAscend> {
#ifdef INFINI_HAS_ATB
  using type = List<0>;
#else
  using type = List<>;
#endif
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_REGISTRY_H_
