#ifndef INFINI_OPS_METAX_TRIL_KERNEL_H_
#define INFINI_OPS_METAX_TRIL_KERNEL_H_

#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"

namespace infini::ops {

template <>
class Operator<Tril, Device::Type::kMetax>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kMetax>,
                                               Tril, false> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kMetax>, Tril,
                                          false>::CudaTriangular;
};

}  // namespace infini::ops

#endif
