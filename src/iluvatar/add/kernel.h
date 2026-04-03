#ifndef INFINI_OPS_ILUVATAR_ADD_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "iluvatar/caster_.h"
#include "iluvatar/data_type_.h"
#include "iluvatar/device_property.h"

namespace infini::ops {

namespace add {

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

}  // namespace add

template <>
class Operator<Add, Device::Type::kIluvatar>
    : public CudaAdd<add::IluvatarBackend> {
 public:
  using CudaAdd<add::IluvatarBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
