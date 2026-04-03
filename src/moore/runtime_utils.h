#ifndef INFINI_OPS_MOORE_RUNTIME_UTILS_H_
#define INFINI_OPS_MOORE_RUNTIME_UTILS_H_

#include "cuda/runtime_utils.h"
#include "moore/device_.h"

namespace infini::ops {

template <>
struct RuntimeUtils<Device::Type::kMoore> {
  static int GetOptimalBlockSize() {
    int max_threads = QueryMaxThreadsPerBlock();
    if (max_threads >= 2048) return 2048;
    if (max_threads >= 1024) return 1024;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
  }
};

}  // namespace infini::ops

#endif
