#ifndef INFINI_OPS_CAMBRICON_NEG_KERNEL_H_
#define INFINI_OPS_CAMBRICON_NEG_KERNEL_H_

#include "base/neg.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Neg, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Neg, cnnlNegTensor> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
