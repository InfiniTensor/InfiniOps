#ifndef INFINI_OPS_METAX_RUNTIME_UTILS_H_
#define INFINI_OPS_METAX_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "cuda/metax/device_property.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kMetax>
    : CudaRuntimeUtils<QueryMaxThreadsPerBlock> {};

}  // namespace infini::ops

#endif
