#ifndef INFINI_OPS_TORCH_ADD_REGISTRY_H_
#define INFINI_OPS_TORCH_ADD_REGISTRY_H_

#include "base/add.h"

namespace infini::ops {

template <Device::Type kDev>
struct ActiveImplementationsImpl<Add, kDev, 1> {
  using type = List<1>;
};

}  // namespace infini::ops

#endif
