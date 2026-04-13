#ifndef INFINI_OPS_TORCH_GEMM_REGISTRY_H_
#define INFINI_OPS_TORCH_GEMM_REGISTRY_H_

#include "base/gemm.h"

namespace infini::ops {

template <Device::Type kDev>
struct ActiveImplementationsImpl<Gemm, kDev, 1> {
  using type = List<2>;
};

}  // namespace infini::ops

#endif
