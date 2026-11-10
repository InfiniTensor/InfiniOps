#ifndef INFINI_OPS_METAX_LINEAR_KERNEL_H_
#define INFINI_OPS_METAX_LINEAR_KERNEL_H_

#include "native/cuda/metax/ops/add/kernel.h"
#include "native/cuda/metax/ops/gemm/mcblas.h"
#include "native/cuda/ops/linear/kernel.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kMetax>
    : public CudaLinear<Device::Type::kMetax> {
 public:
  using CudaLinear<Device::Type::kMetax>::CudaLinear;
};

}  // namespace infini::ops

#endif
