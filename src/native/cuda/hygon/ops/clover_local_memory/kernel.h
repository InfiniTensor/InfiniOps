#ifndef INFINI_OPS_Hygon_CLOVER_LOCAL_MEMORY_KERNEL_H_
#define INFINI_OPS_Hygon_CLOVER_LOCAL_MEMORY_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_local_memory/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverLocalMemory, Device::Type::kHygon>
    : public CudaCloverLocalMemory<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverLocalMemory<
      Runtime<Device::Type::kHygon>>::CudaCloverLocalMemory;
};

}  // namespace infini::ops

#endif
