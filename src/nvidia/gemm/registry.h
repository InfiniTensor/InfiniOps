#ifndef INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_
#define INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_

#include "base/gemm.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<Gemm, Device::Type::kNvidia, 0> {
  using type = List<0, 1>;
};

}  // namespace infini::ops

#endif
