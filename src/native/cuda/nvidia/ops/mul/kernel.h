#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia>
    : public CudaMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMul<Runtime<Device::Type::kNvidia>>::CudaMul;
  // All dispatched types use kernels and constructor-owned metadata on replay.
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
