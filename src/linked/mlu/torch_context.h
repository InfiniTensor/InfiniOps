#ifndef INFINI_OPS_LINKED_MLU_TORCH_CONTEXT_H_
#define INFINI_OPS_LINKED_MLU_TORCH_CONTEXT_H_

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>

namespace infini::ops::linked::mlu {

class TorchContextGuard {
 public:
  TorchContextGuard(void* stream, int device_index)
      : device_guard_{c10::Device{c10::DeviceType::PrivateUse1,
                                  static_cast<c10::DeviceIndex>(device_index)}},
        stream_guard_{torch_mlu::getStreamFromExternal(
            reinterpret_cast<cnrtQueue_t>(stream),
            static_cast<c10::DeviceIndex>(device_index))} {}

 private:
  c10::DeviceGuard device_guard_;

  c10::StreamGuard stream_guard_;
};

}  // namespace infini::ops::linked::mlu

#endif  // INFINI_OPS_LINKED_MLU_TORCH_CONTEXT_H_
