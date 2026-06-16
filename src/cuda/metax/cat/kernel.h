#ifndef INFINI_OPS_METAX_CAT_KERNEL_H_
#define INFINI_OPS_METAX_CAT_KERNEL_H_

#include <utility>

#include "cuda/cat/kernel.h"
#include "cuda/metax/caster.cuh"
#include "cuda/metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kMetax>
    : public CudaCat<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCat<Runtime<Device::Type::kMetax>>::CudaCat;
};

}  // namespace infini::ops

#endif
