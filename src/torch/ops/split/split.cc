#include "torch/ops/split/split.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Split, kDev, 1>::Operator(const Tensor input, const int64_t split_size,
                                   const int64_t dim, std::vector<Tensor> outs)
    : Split{input, split_size, dim, std::move(outs)},
      device_index_{Split::device_index_} {}

template <Device::Type kDev>
void Operator<Split, kDev, 1>::operator()(const Tensor input,
                                          const int64_t split_size,
                                          const int64_t dim,
                                          std::vector<Tensor> outs) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);

  const auto parts = at::split(at_input, split_size_, dim_);
  assert(parts.size() == outs_.size() && "`Split` part count mismatch");

  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto at_out_i =
        ToAtenTensor<kDev>(outs[i].data(), out_shapes_[i], out_strides_[i],
                           out_types_[i], device_index_);
    at_out_i.copy_(parts[i]);
  }
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Split, 1)

}  // namespace infini::ops
