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
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
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
