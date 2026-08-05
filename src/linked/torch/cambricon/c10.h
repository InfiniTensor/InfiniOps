#ifndef INFINI_OPS_LINKED_TORCH_CAMBRICON_C10_H_
#define INFINI_OPS_LINKED_TORCH_CAMBRICON_C10_H_

#include <c10/core/StreamGuard.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>

#include "linked/torch/c10.h"

namespace infini::ops::linked::torch {

template <>
struct C10<Device::Type::kCambricon> {
  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  using StreamGuard = c10::StreamGuard;

  static auto GetStreamFromExternal(void* stream, int device_index) {
    return torch_mlu::getStreamFromExternal(
        reinterpret_cast<cnrtQueue_t>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_CAMBRICON_C10_H_
