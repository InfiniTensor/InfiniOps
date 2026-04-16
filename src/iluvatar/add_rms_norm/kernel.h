#ifndef INFINI_OPS_ILUVATAR_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/add_rms_norm/kernel.h"
#include "iluvatar/add/kernel.h"
#include "iluvatar/add_rms_norm/registry.h"
#include "iluvatar/caster.cuh"
#include "iluvatar/rms_norm/kernel.h"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kIluvatar, 0>
    : public CudaAddRmsNorm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kIluvatar>>::CudaAddRmsNorm;
};

template <>
class Operator<AddRmsNorm, Device::Type::kIluvatar, 1>
    : public CudaAddRmsNormFused<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaAddRmsNormFused<
      Runtime<Device::Type::kIluvatar>>::CudaAddRmsNormFused;
};

}  // namespace infini::ops

#endif
