#ifndef INFINI_OPS_CPU_MUL_REGISTRY_H_
#define INFINI_OPS_CPU_MUL_REGISTRY_H_

#include "base/mul.h"
#include "impl.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Mul, Device::Type::kCpu> {
  using type = List<Impl::kDefault, Impl::kDsl>;
};

}  // namespace infini::ops

#endif
