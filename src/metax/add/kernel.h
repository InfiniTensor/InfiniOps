#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "metax/caster_.h"
#include "metax/data_type_.h"
#include "metax/device_property.h"

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
    return ComputeOptimalBlockSize(QueryMaxThreadsPerBlock());
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
