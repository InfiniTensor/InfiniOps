#ifndef INFINI_OPS_LINKED_CUDA_TORCH_CONTEXT_H_
#define INFINI_OPS_LINKED_CUDA_TORCH_CONTEXT_H_

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

namespace infini::ops::linked::cuda {

class TorchContextGuard {
 public:
  TorchContextGuard(void* stream, int device_index)
      : device_guard_{static_cast<c10::DeviceIndex>(device_index)},
        stream_guard_{c10::cuda::getStreamFromExternal(
            reinterpret_cast<cudaStream_t>(stream),
            static_cast<c10::DeviceIndex>(device_index))} {}

 private:
  c10::cuda::CUDAGuard device_guard_;

  c10::cuda::CUDAStreamGuard stream_guard_;
};

}  // namespace infini::ops::linked::cuda

#endif  // INFINI_OPS_LINKED_CUDA_TORCH_CONTEXT_H_
