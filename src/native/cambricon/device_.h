#ifndef INFINI_OPS_CAMBRICON_DEVICE__H_
#define INFINI_OPS_CAMBRICON_DEVICE__H_

#include "device.h"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kCambricon> : std::true_type {};

}  // namespace infini::ops

#endif
