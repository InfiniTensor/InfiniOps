#ifndef INFINI_OPS_TORCH_ASCEND_C10_H_
#define INFINI_OPS_TORCH_ASCEND_C10_H_

#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
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
    const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
    TORCH_CHECK(task_queue != nullptr && std::strcmp(task_queue, "0") == 0,
                "Ascend external runtime interop requires TASK_QUEUE_ENABLE=0 "
                "before importing torch_npu.");
    // ACL may have switched devices without updating torch_npu's TLS.
    // Align with the actual entry device, not the requested target, so the
    // stream guard can restore that entry device on success and exceptions.
    int entry_device = 0;
    const auto status = aclrtGetDevice(&entry_device);
    TORCH_CHECK(status == ACL_ERROR_NONE,
                "Cannot query the initialized ACL entry device: ", status);
    const auto set_status = c10_npu::SetDevice(entry_device);
    TORCH_CHECK(
        set_status == ACL_ERROR_NONE,
        "Cannot align torch_npu with the ACL entry device: ", set_status);
    const auto index = static_cast<c10::DeviceIndex>(device_index);
    const auto default_stream = c10_npu::getDefaultNPUStream(index);
    if (stream == nullptr || stream == default_stream.stream(false)) {
      // `torch_npu` rejects null external streams, while InfiniOps uses null
      // to select the default device stream.
      return default_stream;
    }

    // Preserve framework stream identity when its raw handle is passed back.
    // torch_npu events reject an external wrapper even for the same address.
    const auto current_stream = c10_npu::getCurrentNPUStream(index);
    if (stream == current_stream.stream(false)) {
      return current_stream;
    }

    return c10_npu::getStreamFromExternal(reinterpret_cast<aclrtStream>(stream),
                                          index);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ASCEND_C10_H_
