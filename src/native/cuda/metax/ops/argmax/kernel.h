#ifndef INFINI_OPS_METAX_ARGMAX_KERNEL_H_
#define INFINI_OPS_METAX_ARGMAX_KERNEL_H_

#include <utility>

#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/argmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kMetax>
    : public CudaArgmax<Runtime<Device::Type::kMetax>> {
 public:
  using CudaArgmax<Runtime<Device::Type::kMetax>>::CudaArgmax;
};

}  // namespace infini::ops

#endif
