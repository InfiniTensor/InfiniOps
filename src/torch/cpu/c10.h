#ifndef INFINI_OPS_TORCH_CPU_C10_H_
#define INFINI_OPS_TORCH_CPU_C10_H_

#include <cstddef>

#include "torch/c10.h"

namespace infini::ops {

template <>
struct C10<Device::Type::kCpu> {
  struct StreamGuard {
    explicit StreamGuard(std::nullptr_t) {}
  };

  static std::nullptr_t GetStreamFromExternal(void*, int) { return nullptr; }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_CPU_C10_H_
