#ifndef INFINI_OPS_METAX_TRIU_KERNEL_H_
#define INFINI_OPS_METAX_TRIU_KERNEL_H_

#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/triangular/kernel.h"

namespace infini::ops {

template <>
class Operator<Triu, Device::Type::kMetax>
    : public triangular_detail::CudaTriangular<Runtime<Device::Type::kMetax>,
                                               Triu, true> {
 public:
  using triangular_detail::CudaTriangular<Runtime<Device::Type::kMetax>, Triu,
                                          true>::CudaTriangular;
};

}  // namespace infini::ops

#endif
