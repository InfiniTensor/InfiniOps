#ifndef INFINI_OPS_METAX_MROPE_KERNEL_H_
#define INFINI_OPS_METAX_MROPE_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mrope/kernel.h"

namespace infini::ops {

template <>
class Operator<Mrope, Device::Type::kMetax>
    : public CudaMrope<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMrope<Runtime<Device::Type::kMetax>>::CudaMrope;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_MROPE_KERNEL_H_
