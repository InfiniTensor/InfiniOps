#ifndef INFINI_OPS_Nvidia_CLOVER_RECURRENT_GDN_KERNEL_H_
#define INFINI_OPS_Nvidia_CLOVER_RECURRENT_GDN_KERNEL_H_

#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_recurrent_gdn/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverRecurrentGdn, Device::Type::kNvidia>
    : public CudaCloverRecurrentGdn<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCloverRecurrentGdn<
      Runtime<Device::Type::kNvidia>>::CudaCloverRecurrentGdn;
};

}  // namespace infini::ops

#endif
