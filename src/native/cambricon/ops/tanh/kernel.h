#ifndef INFINI_OPS_CAMBRICON_TANH_KERNEL_H_
#define INFINI_OPS_CAMBRICON_TANH_KERNEL_H_

#include "base/tanh.h"
#include "native/cambricon/activation_cnnl.h"

namespace infini::ops {

template <>
class Operator<Tanh, Device::Type::kCambricon>
    : public cnnl_utils::Activation<Tanh, CNNL_ACTIVATION_TANH> {
 public:
  Operator(const Tensor input, Tensor out) : Activation{input, out} {}
};

}  // namespace infini::ops

#endif
