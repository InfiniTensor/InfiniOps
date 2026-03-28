#ifndef INFINI_OPS_NVIDIA_ADD_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "nvidia/device_.h"

namespace infini::ops {

namespace add {

struct NvidiaBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr auto malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = cudaMemcpy;

  static constexpr auto free = cudaFree;

  static constexpr auto memcpyH2D = cudaMemcpyHostToDevice;

  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
};

}  // namespace add

template <>
class Operator<Add, Device::Type::kNvidia>
    : public CudaAdd<add::NvidiaBackend> {
 public:
  using CudaAdd<add::NvidiaBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
