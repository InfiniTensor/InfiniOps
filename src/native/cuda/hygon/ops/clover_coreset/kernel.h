#ifndef INFINI_OPS_HYGON_CLOVER_CORESET_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_CORESET_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_coreset/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverCoreset, Device::Type::kHygon>
    : public CudaCloverCoreset<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverCoreset<Runtime<Device::Type::kHygon>>::CudaCloverCoreset;
};

}  // namespace infini::ops

#endif
