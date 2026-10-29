#ifndef INFINI_OPS_TORCH_EXTERNAL_TENSOR_H_
#define INFINI_OPS_TORCH_EXTERNAL_TENSOR_H_

#include <torch/torch.h>

#include <mutex>

#include "device.h"

namespace infini::ops::detail {

template <Device::Type kDev>
inline void InitializeTorchDevice(const at::TensorOptions& options) {
  // A zero-element factory initializes the registered backend on both old and
  // new Torch versions, without querying ownership of an external pointer.
  static std::once_flag initialized;
  std::call_once(initialized, [&] { (void)at::empty({0}, options); });
}

template <Device::Type kDev>
struct ExternalTorchTensor {
  static at::Tensor Wrap(void* data, at::IntArrayRef shape,
                         at::IntArrayRef strides,
                         const at::TensorOptions& options) {
    return at::for_blob(data, shape)
        .strides(strides)
        .options(options)
        .target_device(options.device())
        .make_tensor();
  }
};

}  // namespace infini::ops::detail

#endif  // INFINI_OPS_TORCH_EXTERNAL_TENSOR_H_
