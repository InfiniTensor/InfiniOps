#ifndef INFINI_OPS_METAX_ADD_KERNEL_H_
#define INFINI_OPS_METAX_ADD_KERNEL_H_

#include <utility>

#include "cuda/add/kernel.h"
#include "metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMetax>
    : public CudaAdd<Runtime<Device::Type::kMetax>> {
 public:
  using CudaAdd<Runtime<Device::Type::kMetax>>::CudaAdd;
};

}  // namespace infini::ops

#endif
