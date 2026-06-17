#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include <infini/rt/metax/runtime_.h>
#include "native/cuda/metax/runtime_utils.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMetax>
    : public CudaAdd<Runtime<Device::Type::kMetax>> {
 public:
  using CudaAdd<Runtime<Device::Type::kMetax>>::CudaAdd;
};

}  // namespace infini::ops

#endif
