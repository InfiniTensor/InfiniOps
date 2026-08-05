#ifndef INFINI_OPS_LINKED_TORCH_CONTEXT_H_
#define INFINI_OPS_LINKED_TORCH_CONTEXT_H_

#include "device.h"

namespace infini::ops::linked {

template <Device::Type kDev>
struct TorchStreamBridge;

template <Device::Type kDev>
class TorchContextGuard {
 public:
  using Bridge = TorchStreamBridge<kDev>;
  using Guard = typename Bridge::Guard;

  TorchContextGuard(void* stream, int device_index)
      : stream_guard_{Bridge::FromExternal(stream, device_index)} {}

 private:
  Guard stream_guard_;
};

}  // namespace infini::ops::linked

#endif  // INFINI_OPS_LINKED_TORCH_CONTEXT_H_
