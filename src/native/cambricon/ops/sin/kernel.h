#ifndef INFINI_OPS_CAMBRICON_SIN_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SIN_KERNEL_H_

#include "base/sin.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Sin, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Sin, cnnlSin> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
