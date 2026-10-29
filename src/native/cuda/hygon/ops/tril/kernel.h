#ifndef INFINI_OPS_HYGON_TRIL_KERNEL_H_
#define INFINI_OPS_HYGON_TRIL_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"

namespace infini::ops {

template <>
class Operator<Tril, Device::Type::kHygon>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kHygon>,
                                               Tril, false> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kHygon>, Tril,
                                          false>::CudaTriangular;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_TRIL_KERNEL_H_
