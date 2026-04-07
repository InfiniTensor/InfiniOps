#ifndef INFINI_OPS_METAX_DEVICE__H_
#define INFINI_OPS_METAX_DEVICE__H_

#include "device.h"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kMetax> : std::true_type {};

}  // namespace infini::ops

#endif
