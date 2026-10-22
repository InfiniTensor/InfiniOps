#ifndef INFINI_OPS_TORCH_ASCEND_INTEROP_H_
#define INFINI_OPS_TORCH_ASCEND_INTEROP_H_

#include <torch_npu/csrc/aten/common/from_blob.h>

#include "torch/ascend/c10.h"
#include "torch/interop_base.h"

namespace infini::ops {

template <>
struct TorchInterop<Device::Type::kAscend>
    : DefaultTorchInterop<Device::Type::kAscend> {
  static at::Tensor WrapExternalTensor(void* data, at::IntArrayRef shape,
                                       at::IntArrayRef strides,
                                       const at::TensorOptions& options) {
    // External plain strided storage needs an NPU storage descriptor.
    return at_npu::native::from_blob(data, shape, strides, 0, options,
                                     options.device());
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_ASCEND_INTEROP_H_
