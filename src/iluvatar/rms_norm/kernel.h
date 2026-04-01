#ifndef INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_

#include <utility>

#include "cuda/rms_norm/kernel.h"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kIluvatar>
    : public CudaRmsNorm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kIluvatar>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
