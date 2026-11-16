#ifndef INFINI_OPS_CAMBRICON_SIGN_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SIGN_KERNEL_H_

#include "base/sign.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Sign, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Sign, cnnlSign> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
