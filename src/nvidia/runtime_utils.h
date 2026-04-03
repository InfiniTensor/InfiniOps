#ifndef INFINI_OPS_NVIDIA_RUNTIME_UTILS_H_
#define INFINI_OPS_NVIDIA_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "nvidia/device_property.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kNvidia>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
