#ifndef INFINI_OPS_ASCEND_SWIGLU_REGISTRY_H_
#define INFINI_OPS_ASCEND_SWIGLU_REGISTRY_H_

#include "base/swiglu.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Swiglu, Device::Type::kAscend> {
  using type = List<0, 1>;
};

}  // namespace infini::ops

#endif
