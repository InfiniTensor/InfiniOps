#ifndef INFINI_OPS_TORCH_STREAM_GUARD_H_
#define INFINI_OPS_TORCH_STREAM_GUARD_H_

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
#ifdef WITH_METAX
#include "torch/metax/c10.h"
#endif
#ifdef WITH_MOORE
#include "torch/moore/c10.h"
#endif
#ifdef WITH_HYGON
#include "torch/hygon/c10.h"
#endif
#ifdef WITH_ILUVATAR
#include "torch/iluvatar/c10.h"
#endif
#ifdef WITH_THEAD
#include "torch/thead/c10.h"
#endif
#ifdef WITH_MARS
#include "torch/mars/c10.h"
#endif

namespace infini::ops {
template <Device::Type kDev, typename Fn>
void WithTorchStream(void* stream, int device_index, Fn&& fn) {
  if constexpr (kDev == Device::Type::kNvidia ||
                kDev == Device::Type::kCambricon ||
                kDev == Device::Type::kAscend || kDev == Device::Type::kMetax ||
                kDev == Device::Type::kMoore || kDev == Device::Type::kHygon ||
                kDev == Device::Type::kIluvatar ||
                kDev == Device::Type::kThead || kDev == Device::Type::kMars) {
    const typename C10<kDev>::StreamGuard guard{
        C10<kDev>::GetStreamFromExternal(stream, device_index)};
    fn();
  } else {
    fn();
  }
}
}  // namespace infini::ops

#endif
