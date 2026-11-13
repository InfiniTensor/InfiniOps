#ifndef INFINI_OPS_CAMBRICON_SQRT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SQRT_KERNEL_H_

#include "base/sqrt.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Sqrt, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Sqrt, cnnlSqrt> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
