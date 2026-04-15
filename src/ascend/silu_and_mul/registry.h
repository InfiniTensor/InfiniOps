#ifndef INFINI_OPS_ASCEND_SILU_AND_MUL_REGISTRY_H_
#define INFINI_OPS_ASCEND_SILU_AND_MUL_REGISTRY_H_

#include "base/silu_and_mul.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<SiluAndMul, Device::Type::kAscend> {
  using type = List<0>;
};

}  // namespace infini::ops

#endif
