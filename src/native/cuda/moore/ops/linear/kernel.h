#ifndef INFINI_OPS_MOORE_LINEAR_KERNEL_H_
#define INFINI_OPS_MOORE_LINEAR_KERNEL_H_

#include "native/cuda/moore/ops/add/kernel.h"
#include "native/cuda/moore/ops/gemm/mublas.h"
#include "native/cuda/ops/linear/kernel.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kMoore>
    : public CudaLinear<Device::Type::kMoore> {
 public:
  using CudaLinear<Device::Type::kMoore>::CudaLinear;
};

}  // namespace infini::ops

#endif
