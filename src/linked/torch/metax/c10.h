#ifndef INFINI_OPS_LINKED_TORCH_METAX_C10_H_
#define INFINI_OPS_LINKED_TORCH_METAX_C10_H_

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include "linked/torch/c10.h"

namespace infini::ops::linked::torch {

template <>
struct C10<Device::Type::kMetax> {
  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  using StreamGuard = c10::cuda::CUDAStreamGuard;

  static c10::cuda::CUDAStream GetStreamFromExternal(void* stream,
                                                     int device_index) {
    return c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_METAX_C10_H_
