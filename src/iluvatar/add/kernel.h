#ifndef INFINI_OPS_ILUVATAR_ADD_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "iluvatar/caster.cuh"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kIluvatar>
    : public CudaAdd<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaAdd<Runtime<Device::Type::kIluvatar>>::CudaAdd;
};

}  // namespace infini::ops

#endif
