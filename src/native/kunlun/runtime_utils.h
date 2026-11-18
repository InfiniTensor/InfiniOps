#ifndef INFINI_OPS_KUNLUN_RUNTIME_UTILS_H_
#define INFINI_OPS_KUNLUN_RUNTIME_UTILS_H_

#include "native/cuda/runtime_utils.h"

namespace infini::ops {

// XPU3 clusters run 64 cores; the elementwise kernels cap the grid-sized
// block dimension at 64.
template <>
struct RuntimeUtils<Device::Type::kKunlun> {
  static int GetOptimalBlockSize() { return 64; }
};

}  // namespace infini::ops

#endif
