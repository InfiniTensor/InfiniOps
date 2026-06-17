#ifndef INFINI_OPS_METAX_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_CONV_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include <infini/rt/metax/runtime_.h>
#include "native/cuda/metax/runtime_utils.h"
#include "native/cuda/ops/conv_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ConvInfinilm, Device::Type::kMetax>
    : public CudaConvInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaConvInfinilm<Runtime<Device::Type::kMetax>>::CudaConvInfinilm;
};

}  // namespace infini::ops

#endif
