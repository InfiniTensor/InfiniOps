#ifndef INFINI_OPS_CAMBRICON_COS_KERNEL_H_
#define INFINI_OPS_CAMBRICON_COS_KERNEL_H_

#include "base/cos.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Cos, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Cos, cnnlCos> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
