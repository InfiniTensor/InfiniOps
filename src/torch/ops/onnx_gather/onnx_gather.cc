#include "torch/ops/onnx_gather/onnx_gather.h"
#include "torch/tensor_.h"
#include "torch/encoder_stream.h"

namespace infini::ops {
template <Device::Type kDev>
void Operator<OnnxGather, kDev, 8>::operator()(const Tensor input,
    const Tensor index, const int64_t axis, Tensor out) const {
  WithEncoderStream<kDev>(stream_, out.device().index(), [&] {
    auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input.shape(),
        input.strides(), input.dtype(), input.device().index());
    auto indices = ToAtenTensor<kDev>(const_cast<void*>(index.data()), index.shape(),
        index.strides(), index.dtype(), index.device().index()).reshape({-1});
    auto y = ToAtenTensor<kDev>(out.data(), out.shape(), out.strides(),
        out.dtype(), out.device().index());
    // Normalize only negative indices. Out-of-range indices remain invalid.
    auto normalized = at::where(indices < 0, indices + x.size(axis), indices);
    auto flat_shape = x.sizes().vec();
    flat_shape[axis] = indices.numel();
    auto flat_out = y.view(flat_shape);
    at::index_select_out(flat_out, x, axis, normalized);
  });
}
#ifdef WITH_CPU
template class Operator<OnnxGather, Device::Type::kCpu, 8>;
#endif
#ifdef WITH_NVIDIA
template class Operator<OnnxGather, Device::Type::kNvidia, 8>;
#endif
#ifdef WITH_CAMBRICON
template class Operator<OnnxGather, Device::Type::kCambricon, 8>;
#endif
#ifdef WITH_ASCEND
template class Operator<OnnxGather, Device::Type::kAscend, 8>;
#endif
}
