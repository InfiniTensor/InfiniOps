#ifndef INFINI_OPS_NVIDIA_MATMUL_REGISTRY_H_
#define INFINI_OPS_NVIDIA_MATMUL_REGISTRY_H_

#include "base/matmul.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Matmul, Device::Type::kNvidia> {
  using type = List<0, 1>;
};

}  // namespace infini::ops

#endif
