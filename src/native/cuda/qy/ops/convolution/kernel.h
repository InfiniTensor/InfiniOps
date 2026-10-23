#ifndef INFINI_OPS_QY_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_QY_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/ops/convolution/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kQy>
    : public CudaConv<Runtime<Device::Type::kQy>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kQy>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
