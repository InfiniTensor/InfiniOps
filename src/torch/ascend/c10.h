#ifndef INFINI_OPS_TORCH_ASCEND_C10_H_
#define INFINI_OPS_TORCH_ASCEND_C10_H_

#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdlib>
#include <cstring>

#include "torch/c10.h"

namespace infini::ops {

template <>
struct C10<Device::Type::kAscend> {
  static constexpr Device::Type kDeviceType = Device::Type::kAscend;
  using StreamGuard = c10_npu::NPUStreamGuard;

  static c10_npu::NPUStream GetStreamFromExternal(void* stream,
                                                  int device_index) {
    const auto device = static_cast<c10::DeviceIndex>(device_index);
    if (stream == nullptr) {
      return c10_npu::getCurrentNPUStream(device);
    }
    // A host-queued ATen launch can be overtaken by a native ACL copy or
    // EndCapture on this external stream, even after querying stream().
    // OptionsManager's getter is not exported by torch_npu wheels. Require
    // the documented startup setting rather than linking to that private API.
    const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
    TORCH_CHECK(task_queue != nullptr && std::strcmp(task_queue, "0") == 0,
                "Ascend ATen external streams require TASK_QUEUE_ENABLE=0 "
                "before importing torch_npu. Its asynchronous host task queue "
                "does not preserve ordering with external runtime copies "
                "and graph capture.");
    return c10_npu::getStreamFromExternal(stream, device);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ASCEND_C10_H_
