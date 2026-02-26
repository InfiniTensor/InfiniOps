#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr auto malloc = mcMalloc;
  static constexpr auto memcpy = mcMemcpy;
  static constexpr auto free = mcFree;
  static constexpr auto MemcpyH2D = mcMemcpyHostToDevice;
};

template <>
class Operator<Add, Device::Type::kMetax> : public CudaAdd<MetaxBackend> {
 public:
  using CudaAdd<MetaxBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
