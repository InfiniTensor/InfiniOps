#ifndef INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "iluvatar/caster.cuh"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kIluvatar>
    : public CudaSwiglu<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kIluvatar>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
