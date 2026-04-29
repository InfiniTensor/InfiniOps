#ifndef INFINI_OPS_NVIDIA_DEVICE__H_
#define INFINI_OPS_NVIDIA_DEVICE__H_

#include "device.h"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kNvidia> : std::true_type {};

}  // namespace infini::ops

#endif
