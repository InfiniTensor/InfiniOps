#ifndef INFINI_OPS_CAMBRICON_LESS_KERNEL_H_
#define INFINI_OPS_CAMBRICON_LESS_KERNEL_H_

#include "base/less.h"
#include "native/cambricon/compare_cnnl.h"

namespace infini::ops {

template <>
class Operator<Less, Device::Type::kCambricon>
    : public cnnl_utils::Compare<Less, CNNL_LOGIC_OP_LT> {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Compare{input, other, out} {}
  Operator(const Tensor input, const double other, Tensor out)
      : Compare{input, other, out} {}
  Operator(Tensor input, const Tensor other) : Compare{input, other} {}
  Operator(Tensor input, const double other) : Compare{input, other} {}
};

}  // namespace infini::ops

#endif
