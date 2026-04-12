#ifndef INFINI_OPS_ASCEND_DEVICE__H_
#define INFINI_OPS_ASCEND_DEVICE__H_

// NOTE: Cannot use `#include "device.h"` here — GCC resolves quoted includes
// relative to the current file first, and `src/ascend/` used to contain a
// `device.h`.  Use `data_type.h` which transitively pulls in `src/device.h`.
#include "data_type.h"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kAscend> : std::true_type {};

}  // namespace infini::ops

#endif
