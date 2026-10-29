#ifndef INFINI_OPS_HYGON_TRIU_KERNEL_H_
#define INFINI_OPS_HYGON_TRIU_KERNEL_H_

#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"

namespace infini::ops {

template <>
class Operator<Triu, Device::Type::kHygon>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kHygon>,
                                               Triu, true> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kHygon>, Triu,
                                          true>::CudaTriangular;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_TRIU_KERNEL_H_
