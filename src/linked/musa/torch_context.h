#ifndef INFINI_OPS_LINKED_MUSA_TORCH_CONTEXT_H_
#define INFINI_OPS_LINKED_MUSA_TORCH_CONTEXT_H_

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/musa/MUSAMacros.h>
#include <c10/musa/MUSAStream.h>
#include <musa_runtime_api.h>

namespace infini::ops::linked::musa {

class TorchContextGuard {
 public:
  TorchContextGuard(void* stream, int device_index)
      : device_guard_{c10::Device{c10::DeviceType::PrivateUse1,
                                  static_cast<c10::DeviceIndex>(device_index)}},
        stream_guard_{c10::musa::getStreamFromExternal(
            reinterpret_cast<musaStream_t>(stream),
            static_cast<c10::DeviceIndex>(device_index))} {}

 private:
  c10::DeviceGuard device_guard_;

  c10::StreamGuard stream_guard_;
};

}  // namespace infini::ops::linked::musa

#endif  // INFINI_OPS_LINKED_MUSA_TORCH_CONTEXT_H_
