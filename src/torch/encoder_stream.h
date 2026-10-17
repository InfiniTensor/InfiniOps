#pragma once
#include "torch/c10.h"
#ifdef WITH_NVIDIA
#include "torch/nvidia/c10.h"
#endif
#ifdef WITH_CAMBRICON
#include "torch/cambricon/c10.h"
#endif
#ifdef WITH_ASCEND
#include "torch/ascend/c10.h"
#endif

namespace infini::ops {
template <Device::Type kDev, typename Fn>
void WithEncoderStream(void* stream, int device_index, Fn&& fn) {
  if constexpr (kDev == Device::Type::kNvidia || kDev == Device::Type::kCambricon || kDev == Device::Type::kAscend) {
    const typename C10<kDev>::StreamGuard guard{
        C10<kDev>::GetStreamFromExternal(stream, device_index)};
    fn();
  } else {
    fn();
  }
}
}
