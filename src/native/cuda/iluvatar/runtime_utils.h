#ifndef INFINI_OPS_ILUVATAR_RUNTIME_UTILS_H_
#define INFINI_OPS_ILUVATAR_RUNTIME_UTILS_H_

#include "cuda/iluvatar/device_property.h"
#include "cuda/runtime_utils.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kIluvatar>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
