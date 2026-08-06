#ifndef INFINI_OPS_LINKED_TORCH_METAX_TORCH_CONTEXT_H_
#define INFINI_OPS_LINKED_TORCH_METAX_TORCH_CONTEXT_H_

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include "linked/torch_context.h"

namespace infini::ops::linked {

template <>
struct TorchStreamBridge<Device::Type::kMetax> {
  using Guard = c10::cuda::CUDAStreamGuard;

  static c10::cuda::CUDAStream FromExternal(void* stream, int device_index) {
    return c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream),
        static_cast<c10::DeviceIndex>(device_index));
  }
};

}  // namespace infini::ops::linked

#endif  // INFINI_OPS_LINKED_TORCH_METAX_TORCH_CONTEXT_H_
