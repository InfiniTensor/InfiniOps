#ifndef INFINI_OPS_METAX_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/add_rms_norm/kernel.h"
#include "metax/add/kernel.h"
#include "metax/add_rms_norm/registry.h"
#include "metax/caster.cuh"
#include "metax/rms_norm/kernel.h"
#include "metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kMetax, 0>
    : public CudaAddRmsNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kMetax>>::CudaAddRmsNorm;
};

template <>
class Operator<AddRmsNorm, Device::Type::kMetax, 1>
    : public CudaAddRmsNormFused<Runtime<Device::Type::kMetax>> {
 public:
  using CudaAddRmsNormFused<Runtime<Device::Type::kMetax>>::CudaAddRmsNormFused;
};

}  // namespace infini::ops

#endif
