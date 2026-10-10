#ifndef INFINI_OPS_QY_RUNTIME_UTILS_H_
#define INFINI_OPS_QY_RUNTIME_UTILS_H_

#include "native/cuda/runtime_utils.h"
#include "native/cuda/qy/device_property.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kQy>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
