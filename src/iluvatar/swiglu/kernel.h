#ifndef INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "iluvatar/data_type_.h"
#include "iluvatar/device_property.h"

namespace infini::ops {

namespace swiglu {

struct IluvatarBackend {
  using stream_t = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

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
class Operator<Swiglu, Device::Type::kIluvatar>
    : public CudaSwiglu<swiglu::IluvatarBackend> {
 public:
  using CudaSwiglu<swiglu::IluvatarBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
