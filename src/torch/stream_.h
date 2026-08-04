#ifndef INFINI_OPS_TORCH_STREAM__H_
#define INFINI_OPS_TORCH_STREAM__H_

#ifdef WITH_NVIDIA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

#include "device.h"

namespace infini::ops::detail {

template <Device::Type kDev>
class TorchStreamGuard {
 public:
  TorchStreamGuard(void*, int) {}
};

#ifdef WITH_NVIDIA
template <>
class TorchStreamGuard<Device::Type::kNvidia> {
 public:
  TorchStreamGuard(void* stream, int device_index)
      : device_guard_{static_cast<c10::DeviceIndex>(device_index)},
        stream_guard_{c10::cuda::getStreamFromExternal(
            reinterpret_cast<cudaStream_t>(stream),
            static_cast<c10::DeviceIndex>(device_index))} {}

 private:
  c10::cuda::CUDAGuard device_guard_;

  c10::cuda::CUDAStreamGuard stream_guard_;
};
#endif

}  // namespace infini::ops::detail

#endif
