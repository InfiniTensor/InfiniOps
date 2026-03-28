#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "metax/device_.h"

namespace infini::ops {

namespace add {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr auto malloc = mcMalloc;

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

}  // namespace add

template <>
class Operator<Add, Device::Type::kMetax> : public CudaAdd<add::MetaxBackend> {
 public:
  using CudaAdd<add::MetaxBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
