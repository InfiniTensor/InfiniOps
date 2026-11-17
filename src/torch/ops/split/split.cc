#include "torch/ops/split/split.h"

#include "torch/instantiate.h"
#include "torch/storage_overlap.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Split, kDev, 1>::Operator(const Tensor input, const int64_t split_size,
                                   const int64_t dim, std::vector<Tensor> outs)
    : Split{input, split_size, dim, std::move(outs)} {}

template <Device::Type kDev>
void Operator<Split, kDev, 1>::operator()(const Tensor input,
                                          const int64_t split_size,
                                          const int64_t dim,
                                          std::vector<Tensor> outs) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);

  // Preserve every source slice before any output can overwrite it.
  const auto parts = at::split(at_input.clone(), split_size_, dim_);
  std::vector<at::Tensor> outputs;
  for (std::size_t i = 0; i < outs.size(); ++i) {
    outputs.push_back(ToAtenTensor<kDev>(outs[i].data(), out_shapes_[i],
                                         out_strides_[i], out_types_[i],
                                         device_index_));
    for (std::size_t j = 0; j < i; ++j) {
      assert(!StorageSpansOverlap(outputs[i], outputs[j]) &&
             "Output storage must not overlap.");
    }
  }
  assert(parts.size() == outs_.size() && "`Split` part count mismatch");

  for (std::size_t i = 0; i < parts.size(); ++i) {
    outputs[i].copy_(parts[i]);
  }
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Split, 1)

}  // namespace infini::ops
