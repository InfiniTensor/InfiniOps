#ifndef INFINI_OPS_METAX_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_METAX_LAYER_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/layer_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<LayerNorm, Device::Type::kMetax, 0>
    : public CudaLayerNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaLayerNorm<Runtime<Device::Type::kMetax>>::CudaLayerNorm;
};

}  // namespace infini::ops

#endif
