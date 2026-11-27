#ifndef INFINI_OPS_HYGON_CAST_KERNEL_H_
#define INFINI_OPS_HYGON_CAST_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/cast/kernel.h"

namespace infini::ops {

// Clover reads gated-delta-net projections through strided views.
template <>
class Operator<Cast, Device::Type::kHygon>
    : public CudaCast<Runtime<Device::Type::kHygon>, /*kStridedLayouts=*/true> {
 public:
  using CudaCast<Runtime<Device::Type::kHygon>, true>::CudaCast;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CAST_KERNEL_H_
