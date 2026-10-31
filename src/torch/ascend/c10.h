#ifndef INFINI_OPS_TORCH_ASCEND_C10_H_
#define INFINI_OPS_TORCH_ASCEND_C10_H_

#include <Python.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "torch/c10.h"

// Exported torch_npu op-plugin API, declared in op_api_common_base.h.
// OptionsManager::GetTaskQueueEnable itself is hidden in some binary wheels.
uint32_t OpApiGetTaskQueueEnable();

namespace infini::ops {

template <>
struct C10<Device::Type::kAscend> {
  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  using StreamGuard = c10_npu::NPUStreamGuard;

  static c10_npu::NPUStream GetStreamFromExternal(void* stream,
                                                  int device_index) {
    TORCH_CHECK(Py_IsInitialized(),
                "Ascend ATen provider requires an initialized Python "
                "interpreter and imported torch_npu module");
    const auto device = static_cast<c10::DeviceIndex>(device_index);
    // InfiniRT activates ACL directly; align torch_npu's cached device first.
    int32_t current = -1;
    TORCH_CHECK(aclrtGetDevice(&current) == ACL_ERROR_NONE &&
                    c10_npu::SetDevice(current) == ACL_ERROR_NONE,
                "Failed to synchronize the current ACL and Torch NPU device");
    // Keep Torch-owned streams in their original queue domain. Rewrapping
    // them as external streams loses the backend's host-queue association.
    const auto default_stream = c10_npu::getDefaultNPUStream(device);
    if (stream == nullptr || stream == default_stream.stream()) {
      return default_stream;
    }
    const auto external_stream = c10_npu::getStreamFromExternal(
        reinterpret_cast<aclrtStream>(stream), device);
    const auto current_stream = c10_npu::getCurrentNPUStream(device);
    // External wrapping preserves the identity of an already external stream.
    if (current_stream.id() != external_stream.id() &&
        stream == current_stream.stream()) {
      return current_stream;
    }
    // Native runtime submissions cannot observe Torch's pending host tasks.
    TORCH_CHECK(OpApiGetTaskQueueEnable() == 0,
                "Ascend external streams require TASK_QUEUE_ENABLE=0 before "
                "importing torch_npu");
    return external_stream;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ASCEND_C10_H_
