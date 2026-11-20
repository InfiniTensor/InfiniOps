#ifndef INFINI_OPS_Hygon_CLOVER_RECURRENT_GDN_KERNEL_H_
#define INFINI_OPS_Hygon_CLOVER_RECURRENT_GDN_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_recurrent_gdn/kernel.h"

namespace infini::ops {

template <>
class Operator<CloverRecurrentGdn, Device::Type::kHygon>
    : public CudaCloverRecurrentGdn<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCloverRecurrentGdn<
      Runtime<Device::Type::kHygon>>::CudaCloverRecurrentGdn;
};

}  // namespace infini::ops

#endif
