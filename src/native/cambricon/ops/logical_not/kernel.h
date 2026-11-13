#ifndef INFINI_OPS_CAMBRICON_LOGICAL_NOT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_LOGICAL_NOT_KERNEL_H_

#include "base/logical_not.h"
#include "native/cambricon/unary_cnnl.h"

namespace infini::ops {

template <>
class Operator<LogicalNot, Device::Type::kCambricon>
    : public cnnl_utils::Unary<LogicalNot, cnnlLogicOpNot, true> {
 public:
  Operator(const Tensor input, Tensor out) : Unary{input, out} {}
};

}  // namespace infini::ops

#endif
