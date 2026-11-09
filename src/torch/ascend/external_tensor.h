#ifndef INFINI_OPS_TORCH_ASCEND_EXTERNAL_TENSOR_H_
#define INFINI_OPS_TORCH_ASCEND_EXTERNAL_TENSOR_H_

#include <torch_npu/csrc/aten/common/from_blob.h>

#include "torch/external_tensor.h"

namespace infini::ops::detail {

// Ascend from_blob performs lazy initialization itself, including GIL handling.
// Do not hold a second initialization lock while entering the vendor factory.
template <>
inline void InitializeTorchDevice<Device::Type::kAscend>(
    const at::TensorOptions&) {}

template <>
struct ExternalTorchTensor<Device::Type::kAscend> {
  static at::Tensor Wrap(void* data, at::IntArrayRef shape,
                         at::IntArrayRef strides,
                         const at::TensorOptions& options) {
    // External plain strided memory needs torch_npu's storage descriptor.
    auto tensor = at_npu::native::from_blob(data, shape, strides, 0, options,
                                            options.device());
    tensor.storage().unsafeGetStorageImpl()->set_resizable(false);
    return tensor;
  }
};

}  // namespace infini::ops::detail

#endif  // INFINI_OPS_TORCH_ASCEND_EXTERNAL_TENSOR_H_
