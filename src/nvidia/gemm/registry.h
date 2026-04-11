#ifndef INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_
#define INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_

#include "base/gemm.h"

namespace infini::ops {

// Gemm-specific implementation indices (both hand-written, not DSL).
struct GemmImpl {
  static constexpr std::size_t kCublas = 0;
  static constexpr std::size_t kCublasLt = 1;
};

template <>
struct ActiveImplementationsImpl<Gemm, Device::Type::kNvidia> {
  using type = List<GemmImpl::kCublas, GemmImpl::kCublasLt>;
};

}  // namespace infini::ops

#endif
