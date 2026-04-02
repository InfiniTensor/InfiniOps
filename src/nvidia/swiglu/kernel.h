#ifndef INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_
#define INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "nvidia/data_type_.h"
#include "nvidia/device_property.h"

namespace infini::ops {

namespace swiglu {

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
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
};

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kNvidia>
    : public CudaSwiglu<swiglu::NvidiaBackend> {
 public:
  using CudaSwiglu<swiglu::NvidiaBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
