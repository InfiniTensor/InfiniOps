#ifndef INFINI_OPS_METAX_MOE_FUSED_DENSE_KERNEL_H_
#define INFINI_OPS_METAX_MOE_FUSED_DENSE_KERNEL_H_

#include <utility>

#include "native/cuda/metax/blas.h"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/moe_fused_dense/kernel.h"

namespace infini::ops {

template <>
class Operator<MoeFusedDense, Device::Type::kMetax>
    : public CudaMoeFusedDense<Blas<Device::Type::kMetax>> {
 public:
  using CudaMoeFusedDense<Blas<Device::Type::kMetax>>::CudaMoeFusedDense;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_MOE_FUSED_DENSE_KERNEL_H_