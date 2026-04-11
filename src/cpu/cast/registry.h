#ifndef INFINI_OPS_CPU_CAST_REGISTRY_H_
#define INFINI_OPS_CPU_CAST_REGISTRY_H_

#include "base/cast.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Cast, Device::Type::kCpu> {
  using type = List<Impl::kDefault, Impl::kDsl>;
};

}  // namespace infini::ops

#endif
