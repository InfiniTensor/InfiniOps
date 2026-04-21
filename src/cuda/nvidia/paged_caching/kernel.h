#ifndef INFINI_OPS_NVIDIA_PAGED_CACHING_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAGED_CACHING_KERNEL_H_

#include "cuda/nvidia/runtime_.h"
#include "cuda/paged_caching/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedCaching, Device::Type::kNvidia>
    : public CudaPagedCaching<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaPagedCaching<Runtime<Device::Type::kNvidia>>::CudaPagedCaching;
};

}  // namespace infini::ops

#endif
