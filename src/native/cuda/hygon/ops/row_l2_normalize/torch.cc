#ifdef WITH_TORCH
#include "native/cuda/hygon/ops/row_l2_normalize/torch.h"

#include "torch/hygon/c10.h"
#include "torch/tensor_.h"
namespace infini::ops {
void Operator<RowL2Normalize, Device::Type::kHygon, 8>::operator()(
    const Tensor input, double eps, Tensor out) const {
  if (!supports(input, eps, out) ||
      !std::equal_to<Tensor>{}(input, input_meta_) ||
      !std::equal_to<Tensor>{}(out, out_meta_) ||
      std::memcmp(&eps, &eps_, sizeof(eps)))
    throw std::invalid_argument(
        "Hygon RowL2Normalize: invalid metadata, eps or alias");
  if (input.numel() == 0) return;
  const C10<Device::Type::kHygon>::StreamGuard guard{
      C10<Device::Type::kHygon>::GetStreamFromExternal(stream_,
                                                       out.device().index())};
  const auto view = [](Tensor t) {
    return ToAtenTensor<Device::Type::kHygon>(const_cast<void*>(t.data()),
                                              t.shape(), t.strides(), t.dtype(),
                                              t.device().index());
  };
  auto x = view(input), y = view(out);
  auto denominator = at::linalg_vector_norm(x, 2, {1}, true).clamp_min(eps);
  at::div_out(y, x, denominator);
}
}  // namespace infini::ops
#endif
