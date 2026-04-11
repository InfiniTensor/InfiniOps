#ifndef INFINI_OPS_NVIDIA_CAST_REGISTRY_H_
#define INFINI_OPS_NVIDIA_CAST_REGISTRY_H_

#include "base/cast.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Cast, Device::Type::kNvidia> {
  using type = List<Impl::kDsl>;
};

}  // namespace infini::ops

#endif
