#ifndef INFINI_OPS_METAX_SWIGLU_KERNEL_H_
#define INFINI_OPS_METAX_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "metax/device_.h"

namespace infini::ops {

namespace swiglu {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto malloc = [](auto&&... args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = mcMemcpy;

  static constexpr auto free = mcFree;

  static constexpr auto memcpyH2D = mcMemcpyHostToDevice;

  static int GetOptimalBlockSize() {
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
  }
};

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kMetax>
    : public CudaSwiglu<swiglu::MetaxBackend> {
 public:
  using CudaSwiglu<swiglu::MetaxBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
