#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#include "native/cuda/nvidia/ops/mul/scalar.h"
#include "native/cuda/ops/mul/scalar_provider.h"

namespace infini::ops {

// The scalar `Mul(tensor, double, tensor)` overload is served by the
// shared CUDA provider skeleton (see scalar_provider.h) with
// NVIDIA's launch-side scalar kernel; the tensor x tensor overload
// comes from `CudaMul` unchanged.
template <>
class Operator<Mul, Device::Type::kNvidia>
    : public CudaScalarMulProvider<Runtime<Device::Type::kNvidia>,
                                   NvidiaScalarMul> {
 public:
  using CudaScalarMulProvider<Runtime<Device::Type::kNvidia>,
                              NvidiaScalarMul>::CudaScalarMulProvider;
};

}  // namespace infini::ops

#endif
