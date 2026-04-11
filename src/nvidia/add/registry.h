#ifndef INFINI_OPS_NVIDIA_ADD_REGISTRY_H_
#define INFINI_OPS_NVIDIA_ADD_REGISTRY_H_

#include "base/add.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Add, Device::Type::kNvidia> {
  using type = List<Impl::kDefault, Impl::kDsl>;
};

}  // namespace infini::ops

#endif
