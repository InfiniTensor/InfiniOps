#ifndef INFINI_OPS_NVIDIA_LINEAR_KERNEL_H_
#define INFINI_OPS_NVIDIA_LINEAR_KERNEL_H_

#include "native/cuda/nvidia/ops/add/kernel.h"
#include "native/cuda/nvidia/ops/gemm/cublas.h"
#include "native/cuda/ops/linear/kernel.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kNvidia>
    : public CudaLinear<Device::Type::kNvidia> {
 public:
  using CudaLinear<Device::Type::kNvidia>::CudaLinear;
};

}  // namespace infini::ops

#endif
