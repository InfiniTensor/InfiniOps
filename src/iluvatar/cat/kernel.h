#ifndef INFINI_OPS_ILUVATAR_CAT_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAT_KERNEL_H_

#include <utility>

#include "cuda/cat/kernel.h"
#include "iluvatar/caster.cuh"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kIluvatar>
    : public CudaCat<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaCat<Runtime<Device::Type::kIluvatar>>::CudaCat;
};

}  // namespace infini::ops

#endif
