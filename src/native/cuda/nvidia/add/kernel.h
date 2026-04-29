#ifndef INFINI_OPS_NVIDIA_ADD_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "cuda/nvidia/caster.cuh"
#include "cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia>
    : public CudaAdd<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAdd<Runtime<Device::Type::kNvidia>>::CudaAdd;
};

}  // namespace infini::ops

#endif
