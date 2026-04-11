#ifndef INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_
#define INFINI_OPS_NVIDIA_GEMM_REGISTRY_H_

#include "base/gemm.h"

namespace infini::ops {

// Gemm-specific implementation indices.
// cuBLAS is the default for stability (matches reference implementations).
// cuBLASLt uses heuristic algorithm selection and is 2-3x faster on
// typical LLM shapes — select with `implementation="cublaslt"`.
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
