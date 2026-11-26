#ifndef INFINI_OPS_CAMBRICON_ERF_KERNEL_H_
#define INFINI_OPS_CAMBRICON_ERF_KERNEL_H_

#include "base/erf.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<Erf, Device::Type::kCambricon>
    : public cnnl_utils::Unary<Erf, cnnlErf> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
