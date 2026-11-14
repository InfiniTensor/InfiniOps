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

#ifdef WITH_ILUVATAR
#include "torch/iluvatar/c10.h"
#endif

#ifdef WITH_HYGON
#include "torch/hygon/c10.h"
#endif

#ifdef WITH_THEAD
#include "torch/thead/c10.h"
#endif

#ifdef WITH_MARS
#include "torch/mars/c10.h"
#endif

#ifdef WITH_QY
#include "torch/qy/c10.h"
#endif

namespace infini::ops {

template <Device::Type kDev>
class TorchStreamGuard {
 public:
  TorchStreamGuard(void* stream, int device_index)
      : guard_{C10<kDev>::GetStreamFromExternal(stream, device_index)} {}

 private:
  typename C10<kDev>::StreamGuard guard_;
};

template <>
class TorchStreamGuard<Device::Type::kCpu> {
 public:
  TorchStreamGuard(void*, int) {}
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_STREAM_GUARD_H_
