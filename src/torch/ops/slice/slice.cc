#include "torch/ops/slice/slice.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Slice, kDev, 1>::Operator(const Tensor input,
                                   const std::vector<int64_t> starts,
                                   const std::vector<int64_t> ends,
                                   const std::vector<int64_t> steps, Tensor out)
    : Slice{input, starts, ends, steps, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Slice, kDev, 1>::operator()(const Tensor input,
                                          const std::vector<int64_t> starts,
                                          const std::vector<int64_t> ends,
                                          const std::vector<int64_t> steps,
                                          Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  using namespace torch::indexing;

  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  std::vector<TensorIndex> index;
  index.reserve(starts_.size());
  for (std::size_t i = 0; i < starts_.size(); ++i) {
    index.emplace_back(at::indexing::Slice(starts_[i], ends_[i], steps_[i]));
  }

  at_out.copy_(at_input.index(index));
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Slice, 1)

}  // namespace infini::ops
