#ifndef INFINI_OPS_NVIDIA_MUL_REGISTRY_H_
#define INFINI_OPS_NVIDIA_MUL_REGISTRY_H_

#include "base/mul.h"
#include "impl.h"

namespace infini::ops {

// Mul has only a DSL implementation on NVIDIA (no hand-written version).
// The dispatcher falls back to the first available implementation when
// the requested index is not found.
template <>
struct ActiveImplementationsImpl<Mul, Device::Type::kNvidia> {
  using type = List<Impl::kDsl>;
};

}  // namespace infini::ops

#endif
