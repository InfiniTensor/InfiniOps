#ifndef INFINI_OPS_HYGON_ADD_KERNEL_H_
#define INFINI_OPS_HYGON_ADD_KERNEL_H_

#include "cuda/add/kernel.h"
#include "hygon/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kHygon>
    : public CudaAdd<Runtime<Device::Type::kHygon>> {
 public:
  using CudaAdd<Runtime<Device::Type::kHygon>>::CudaAdd;
};

}  // namespace infini::ops

#endif
