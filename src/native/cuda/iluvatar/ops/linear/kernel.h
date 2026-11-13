#ifndef INFINI_OPS_ILUVATAR_LINEAR_KERNEL_H_
#define INFINI_OPS_ILUVATAR_LINEAR_KERNEL_H_

#include "native/cuda/iluvatar/ops/add/kernel.h"
#include "native/cuda/iluvatar/ops/gemm/cublas.h"
#include "native/cuda/ops/linear/kernel.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kIluvatar>
    : public CudaLinear<Device::Type::kIluvatar> {
 public:
  using CudaLinear<Device::Type::kIluvatar>::CudaLinear;
};

}  // namespace infini::ops

#endif
