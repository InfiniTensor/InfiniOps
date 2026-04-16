#ifndef INFINI_OPS_ASCEND_ADD_RMS_NORM_REGISTRY_H_
#define INFINI_OPS_ASCEND_ADD_RMS_NORM_REGISTRY_H_

#include "base/add_rms_norm.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<AddRmsNorm, Device::Type::kAscend> {
#ifdef INFINI_HAS_CUSTOM_KERNELS
  using type = List<0, 1, 2>;
#else
  using type = List<0, 1>;
#endif
};

}  // namespace infini::ops

#endif
