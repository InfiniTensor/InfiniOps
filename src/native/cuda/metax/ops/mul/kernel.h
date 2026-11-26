#ifndef INFINI_OPS_METAX_MUL_KERNEL_H_
#define INFINI_OPS_METAX_MUL_KERNEL_H_

#include "native/cuda/metax/ops/mul/scalar.h"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mul/scalar_provider.h"

namespace infini::ops {

// The DeepSeek-V2 MoE router scale (routed_scaling_factor) needs the
// scalar `Mul(tensor, double, tensor)` overload; the wrapper wires
// MetaX's header-only scalar kernel into the shared CUDA provider
// skeleton (see scalar_provider.h). The tensor x tensor overload
// comes from `CudaMul` unchanged.
template <>
class Operator<Mul, Device::Type::kMetax>
    : public CudaScalarMulProvider<Runtime<Device::Type::kMetax>,
                                   MetaxScalarMul> {
 public:
  using CudaScalarMulProvider<Runtime<Device::Type::kMetax>,
                              MetaxScalarMul>::CudaScalarMulProvider;
};

}  // namespace infini::ops

#endif
