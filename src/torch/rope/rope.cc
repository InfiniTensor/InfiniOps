#include "torch/rope/rope.h"

#include "torch/tensor_.h"

#include <ATen/TensorIndexing.h>

namespace infini::ops {

template <Device::Type kDev>
Operator<Rope, kDev, 1>::Operator(const Tensor x, const Tensor positions,
                                  const Tensor sin_cache,
                                  const Tensor cos_cache, bool is_neox_style,
                                  Tensor out)
    : Rope{x,        positions,     sin_cache, cos_cache,
           is_neox_style, out},
      device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Rope, kDev, 1>::operator()(const Tensor x, const Tensor positions,
                                         const Tensor sin_cache,
                                         const Tensor cos_cache,
                                         bool is_neox_style,
                                         Tensor out) const {
  auto at_x = ToAtenTensor<kDev>(const_cast<void*>(x.data()), x_shape_,
                                 x_strides_, x_type_, device_index_);
  auto at_positions = ToAtenTensor<kDev>(
      const_cast<void*>(positions.data()), positions_shape_,
      positions_strides_, positions_type_, device_index_);
  auto at_sin_cache = ToAtenTensor<kDev>(
      const_cast<void*>(sin_cache.data()), sin_cache_shape_,
      sin_cache_strides_, sin_cache_type_, device_index_);
  auto at_cos_cache = ToAtenTensor<kDev>(
      const_cast<void*>(cos_cache.data()), cos_cache_shape_,
      cos_cache_strides_, cos_cache_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  const auto head_dim = at_x.size(-1);
  const auto half = head_dim / 2;

  // Gather sin/cos rows per position.
  // `at_positions` has some shape `S`; `at_sin_pos` / `at_cos_pos` end up
  // `S + [half]`. The rotation needs to broadcast across any middle
  // dimensions of `x` (e.g. heads), so we insert singleton dims to pad
  // `S` out to `x.ndim() - 1`.
  auto positions_long = at_positions.to(at::kLong);
  auto sin_flat = at_sin_cache.index_select(0, positions_long.reshape(-1));
  auto cos_flat = at_cos_cache.index_select(0, positions_long.reshape(-1));
  std::vector<int64_t> freq_shape(positions_long.sizes().begin(),
                                  positions_long.sizes().end());
  freq_shape.push_back(half);
  auto sin_at_pos = sin_flat.view(freq_shape);
  auto cos_at_pos = cos_flat.view(freq_shape);

  const int64_t mid_dims =
      static_cast<int64_t>(at_x.dim()) - 1 - static_cast<int64_t>(at_positions.dim());
  for (int64_t i = 0; i < mid_dims; ++i) {
    sin_at_pos = sin_at_pos.unsqueeze(-2);
    cos_at_pos = cos_at_pos.unsqueeze(-2);
  }

  // Rotation pattern differs between GPT-NeoX (`true`) and GPT-J (`false`)
  // styles. We materialize the two halves, apply the 2x2 rotation, and
  // write the result into `out` via the corresponding view.
  using at::indexing::Slice;
  if (is_neox_style) {
    auto x1 = at_x.index({"...", Slice(0, half)});
    auto x2 = at_x.index({"...", Slice(half, head_dim)});
    auto out1 = at_out.index({"...", Slice(0, half)});
    auto out2 = at_out.index({"...", Slice(half, head_dim)});
    // `mul + sub` into a temporary, then copy, avoids aliasing between
    // `x` and `out` when callers ask for in-place semantics.
    auto rot1 = x1 * cos_at_pos - x2 * sin_at_pos;
    auto rot2 = x2 * cos_at_pos + x1 * sin_at_pos;
    out1.copy_(rot1);
    out2.copy_(rot2);
  } else {
    auto x_even = at_x.index({"...", Slice(0, head_dim, 2)});
    auto x_odd = at_x.index({"...", Slice(1, head_dim, 2)});
    auto out_even = at_out.index({"...", Slice(0, head_dim, 2)});
    auto out_odd = at_out.index({"...", Slice(1, head_dim, 2)});
    auto rot_even = x_even * cos_at_pos - x_odd * sin_at_pos;
    auto rot_odd = x_odd * cos_at_pos + x_even * sin_at_pos;
    out_even.copy_(rot_even);
    out_odd.copy_(rot_odd);
  }
}

template class Operator<Rope, Device::Type::kCpu, 1>;
template class Operator<Rope, Device::Type::kNvidia, 1>;
template class Operator<Rope, Device::Type::kCambricon, 1>;
template class Operator<Rope, Device::Type::kAscend, 1>;
template class Operator<Rope, Device::Type::kMetax, 1>;
template class Operator<Rope, Device::Type::kMoore, 1>;
template class Operator<Rope, Device::Type::kIluvatar, 1>;
template class Operator<Rope, Device::Type::kKunlun, 1>;
template class Operator<Rope, Device::Type::kHygon, 1>;
template class Operator<Rope, Device::Type::kQy, 1>;

}  // namespace infini::ops
