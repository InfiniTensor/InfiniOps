#ifndef INFINI_OPS_CAMBRICON_RSQRT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_RSQRT_KERNEL_H_

#include "base/rsqrt.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Rsqrt, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Rsqrt, cnnlRsqrt> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
