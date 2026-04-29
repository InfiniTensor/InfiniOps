#ifndef INFINI_OPS_NVIDIA_ADD_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_KERNEL_H_

#include <utility>

#include "native/cuda/ops/add/kernel.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia>
    : public CudaAdd<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAdd<Runtime<Device::Type::kNvidia>>::CudaAdd;
};

}  // namespace infini::ops

#endif
