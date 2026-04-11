#ifndef INFINI_OPS_NVIDIA_SWIGLU_REGISTRY_H_
#define INFINI_OPS_NVIDIA_SWIGLU_REGISTRY_H_

#include "base/swiglu.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Swiglu, Device::Type::kNvidia> {
  using type = List<Impl::kDefault, Impl::kDsl>;
};

}  // namespace infini::ops

#endif
