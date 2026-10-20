#include "torch/ops/onnx_gather/onnx_gather.h"

#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {
template <Device::Type kDev>
void Operator<OnnxGather, kDev, 8>::operator()(const Tensor input,
                                               const Tensor index,
                                               const int64_t axis,
                                               Tensor out) const {
  TORCH_CHECK(supports(input, index, axis, out),
              "onnx_gather: unsupported metadata or alias");
  WithTorchStream<kDev>(stream_, out.device().index(), [&] {
    auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input.shape(),
                                input.strides(), input.dtype(),
                                input.device().index());
    auto y = ToAtenTensor<kDev>(out.data(), out.shape(), out.strides(),
                                out.dtype(), out.device().index());
    const int64_t dim = axis < 0 ? axis + x.dim() : axis;
    auto indices = ToAtenTensor<kDev>(const_cast<void*>(index.data()),
                                      index.shape(), index.strides(),
                                      index.dtype(), index.device().index())
                       .reshape({-1});
    // Normalize negative indices exactly once, without copying indices to CPU.
    // Invalid data-dependent indices remain errors in index_select.
    auto normalized = at::where(indices < 0, indices + x.size(dim), indices);
    auto flat_shape = x.sizes().vec();
    flat_shape[dim] = indices.numel();
    if (y.is_contiguous()) {
      auto flat_out = y.view(flat_shape);
      at::index_select_out(flat_out, x, dim, normalized);
    } else {
      y.copy_(at::index_select(x, dim, normalized).reshape(y.sizes()));
    }
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
#ifdef WITH_METAX
template class Operator<OnnxGather, Device::Type::kMetax, 8>;
#endif
#ifdef WITH_MOORE
template class Operator<OnnxGather, Device::Type::kMoore, 8>;
#endif
#ifdef WITH_ILUVATAR
template class Operator<OnnxGather, Device::Type::kIluvatar, 8>;
#endif
#ifdef WITH_HYGON
template class Operator<OnnxGather, Device::Type::kHygon, 8>;
#endif
#ifdef WITH_THEAD
template class Operator<OnnxGather, Device::Type::kThead, 8>;
#endif
#ifdef WITH_MARS
template class Operator<OnnxGather, Device::Type::kMars, 8>;
#endif
}  // namespace infini::ops
