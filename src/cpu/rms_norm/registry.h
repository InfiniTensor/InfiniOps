#ifndef INFINI_OPS_CPU_RMS_NORM_REGISTRY_H_
#define INFINI_OPS_CPU_RMS_NORM_REGISTRY_H_

#include "base/rms_norm.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<RmsNorm, Device::Type::kCpu> {
  using type = List<Impl::kDefault, Impl::kDsl>;
};

}  // namespace infini::ops

#endif
