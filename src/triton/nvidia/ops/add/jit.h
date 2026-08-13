#ifndef INFINI_OPS_TRITON_NVIDIA_OPS_ADD_JIT_H_
#define INFINI_OPS_TRITON_NVIDIA_OPS_ADD_JIT_H_

#include "triton/ops/add/jit.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kNvidia, 10>
    : public triton::jit::Add<Device::Type::kNvidia> {
 public:
  using triton::jit::Add<Device::Type::kNvidia>::Add;
};

}  // namespace infini::ops

#endif
