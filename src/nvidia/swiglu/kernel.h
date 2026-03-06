#ifndef INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_
#define INFINI_OPS_NVIDIA_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kNvidia>
    : public CudaSwiglu<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kNvidia>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
