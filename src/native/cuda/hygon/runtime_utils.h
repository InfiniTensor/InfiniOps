#ifndef INFINI_OPS_HYGON_RUNTIME_UTILS_H_
#define INFINI_OPS_HYGON_RUNTIME_UTILS_H_

#include "native/cuda/hygon/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kHygon>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
