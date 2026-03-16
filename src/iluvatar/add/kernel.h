#ifndef INFINI_OPS_ILUVATAR_ADD_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

namespace add {

struct IluvatarBackend {
  static constexpr auto device_value = Device::Type::kIluvatar;

  using stream_t = cudaStream_t;

  static constexpr auto malloc = [](auto&&... args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = cudaMemcpy;

  static constexpr auto free = cudaFree;

  static constexpr auto memcpyH2D = cudaMemcpyHostToDevice;
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
