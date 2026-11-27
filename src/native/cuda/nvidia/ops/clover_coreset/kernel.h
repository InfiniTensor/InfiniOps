#ifndef INFINI_OPS_NVIDIA_CLOVER_CORESET_KERNEL_H_
#define INFINI_OPS_NVIDIA_CLOVER_CORESET_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_coreset/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverCoreset, Device::Type::kNvidia>
    : public CudaCloverCoreset<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverCoreset<Runtime<Device::Type::kNvidia>>::CudaCloverCoreset;
};

}  // namespace infini::ops

#endif
