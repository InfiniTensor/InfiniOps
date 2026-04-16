#ifndef INFINI_OPS_ASCEND_RMS_NORM_REGISTRY_H_
#define INFINI_OPS_ASCEND_RMS_NORM_REGISTRY_H_

#include "base/rms_norm.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<RmsNorm, Device::Type::kAscend> {
#ifdef INFINI_HAS_CUSTOM_KERNELS
  using type = List<0, 1>;
#else
  using type = List<0>;
#endif
};

}  // namespace infini::ops

#endif
