#ifndef INFINI_OPS_ASCEND_DEVICE__H_
#define INFINI_OPS_ASCEND_DEVICE__H_

#include "device.h"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kAscend> : std::true_type {};

}  // namespace infini::ops

#endif
