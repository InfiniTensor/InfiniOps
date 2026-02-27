#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

namespace add {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr auto malloc = mcMalloc;
  static constexpr auto memcpy = mcMemcpy;
  static constexpr auto free = mcFree;
  static constexpr auto MemcpyH2D = mcMemcpyHostToDevice;
};

}  // namespace add

template <>
class Operator<Add, Device::Type::kMetax> : public CudaAdd<add::MetaxBackend> {
 public:
  using CudaAdd<add::MetaxBackend>::CudaAdd;
};

}  // namespace infini::ops

#endif
