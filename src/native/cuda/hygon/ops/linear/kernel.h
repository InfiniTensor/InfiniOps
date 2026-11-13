#ifndef INFINI_OPS_HYGON_LINEAR_KERNEL_H_
#define INFINI_OPS_HYGON_LINEAR_KERNEL_H_

#include "native/cuda/hygon/ops/add/kernel.h"
#include "native/cuda/hygon/ops/gemm/cublas.h"
#include "native/cuda/ops/linear/kernel.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kHygon>
    : public CudaLinear<Device::Type::kHygon> {
 public:
  using CudaLinear<Device::Type::kHygon>::CudaLinear;
};

}  // namespace infini::ops

#endif
