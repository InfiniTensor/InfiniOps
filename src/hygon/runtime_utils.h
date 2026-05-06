#ifndef INFINI_OPS_HYGON_RUNTIME_UTILS_H_
#define INFINI_OPS_HYGON_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "hygon/device_.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kHygon>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
