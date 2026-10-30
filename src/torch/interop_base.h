#ifndef INFINI_OPS_TORCH_INTEROP_BASE_H_
#define INFINI_OPS_TORCH_INTEROP_BASE_H_

#include <torch/torch.h>

#include <array>
#include <mutex>

#include "device.h"

namespace infini::ops {

namespace detail {

template <typename Context>
auto InitializeTorchDevice(Context& context, const at::TensorOptions& options,
                           int)
    -> decltype(context.lazyInitDevice(options.device().type()), void()) {
  context.lazyInitDevice(options.device().type());
}

template <typename Context>
void InitializeTorchDevice(Context&, const at::TensorOptions& options, long) {
  // Older ATen has no shared lazy-init API. A zero-element factory initializes
  // its registered backend once without querying an external pointer's device.
  static std::array<std::once_flag, c10::COMPILE_TIME_MAX_DEVICE_TYPES>
      initialized;
  const auto type = static_cast<std::size_t>(options.device().type());
  std::call_once(initialized[type], [&] { at::empty({0}, options); });
}

}  // namespace detail

template <Device::Type kDev>
struct TorchInterop {
  static at::Tensor WrapExternalTensor(void* data, at::IntArrayRef shape,
                                       at::IntArrayRef strides,
                                       const at::TensorOptions& options) {
    detail::InitializeTorchDevice(at::globalContext(), options, 0);
    return at::for_blob(data, shape)
        .strides(strides)
        .options(options)
        .target_device(options.device())
        .make_tensor();
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_INTEROP_BASE_H_
