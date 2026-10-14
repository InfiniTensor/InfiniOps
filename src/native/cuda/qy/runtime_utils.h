#ifndef INFINI_OPS_QY_RUNTIME_UTILS_H_
#define INFINI_OPS_QY_RUNTIME_UTILS_H_

#include "native/cuda/qy/device_property.h"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kQy>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
