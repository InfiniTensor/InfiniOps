#ifndef INFINI_OPS_METAX_SWIGLU_KERNEL_H_
#define INFINI_OPS_METAX_SWIGLU_KERNEL_H_

#include <utility>

#include "cuda/swiglu/kernel.h"
#include "cuda/metax/caster.cuh"
#include "cuda/metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kMetax>
    : public CudaSwiglu<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kMetax>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
