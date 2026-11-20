#ifndef INFINI_OPS_Nvidia_CLOVER_LOCAL_MEMORY_KERNEL_H_
#define INFINI_OPS_Nvidia_CLOVER_LOCAL_MEMORY_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_local_memory/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverLocalMemory, Device::Type::kNvidia>
    : public CudaCloverLocalMemory<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverLocalMemory<
      Runtime<Device::Type::kNvidia>>::CudaCloverLocalMemory;
};

}  // namespace infini::ops

#endif
