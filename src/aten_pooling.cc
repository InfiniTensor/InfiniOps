#include <infini/ops/pooling.h>

#ifdef INFINI_OPS_WITH_ASCEND_TORCH_CAPTURE
#include "data_type.h"
#include "torch/ascend/c10.h"
#include "torch/tensor_.h"
#endif

namespace infini::ops {

bool TryMaxPool2dValues(infini::rt::TensorView input,
                        const std::vector<int64_t>& kernel_size,
                        const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, bool ceil_mode,
                        infini::rt::TensorView output, void* stream) {
#ifdef INFINI_OPS_WITH_ASCEND_TORCH_CAPTURE
  if (input.device().type() == Device::Type::kAscend) {
    constexpr auto kDev = Device::Type::kAscend;
    const auto device = input.device().index();
    const C10<kDev>::StreamGuard guard{
        C10<kDev>::GetStreamFromExternal(stream, device)};
    auto at_input = ToAtenTensor<kDev>(input.data(), input.shape(),
                                       input.strides(), input.dtype(), device);
    auto at_output =
        ToAtenTensor<kDev>(output.data(), output.shape(), output.strides(),
                           output.dtype(), device);
    // torch_npu can use a packed INT8 mask rather than ATen's INT64 indices.
    // Let the provider allocate its own representation; the caller only needs
    // values. Both this temporary and the mask belong to the graph's pool.
    auto values = at::max_pool2d(at_input, kernel_size, stride, padding,
                                 dilation, ceil_mode);
    at_output.copy_(values);
    return true;
  }
#else
  (void)input;
  (void)kernel_size;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)ceil_mode;
  (void)output;
  (void)stream;
#endif
  return false;
}

}  // namespace infini::ops
