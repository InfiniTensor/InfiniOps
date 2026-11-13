#ifndef INFINI_OPS_CAMBRICON_ABS_KERNEL_H_
#define INFINI_OPS_CAMBRICON_ABS_KERNEL_H_

#include "base/abs.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Abs, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Abs, cnnlAbs> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
