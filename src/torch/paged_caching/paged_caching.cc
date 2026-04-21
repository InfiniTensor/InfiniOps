#include "torch/paged_caching/paged_caching.h"

#include "torch/tensor_.h"

#include <ATen/TensorIndexing.h>

namespace infini::ops {

template <Device::Type kDev>
Operator<PagedCaching, kDev, 1>::Operator(Tensor k_cache, Tensor v_cache,
                                          const Tensor k, const Tensor v,
                                          const Tensor slot_mapping)
    : PagedCaching{k_cache, v_cache, k, v, slot_mapping},
      device_index_{k_cache.device().index()} {}

template <Device::Type kDev>
void Operator<PagedCaching, kDev, 1>::operator()(
    Tensor k_cache, Tensor v_cache, const Tensor k, const Tensor v,
    const Tensor slot_mapping) const {
  auto at_k_cache = ToAtenTensor<kDev>(k_cache.data(), k_cache_shape_,
                                       k_cache_strides_, k_cache_type_,
                                       device_index_);
  auto at_v_cache = ToAtenTensor<kDev>(v_cache.data(), v_cache_shape_,
                                       v_cache_strides_, v_cache_type_,
                                       device_index_);
  auto at_k = ToAtenTensor<kDev>(const_cast<void*>(k.data()), k_shape_,
                                 k_strides_, k_type_, device_index_);
  auto at_v = ToAtenTensor<kDev>(const_cast<void*>(v.data()), v_shape_,
                                 v_strides_, v_type_, device_index_);
  auto at_slot_mapping =
      ToAtenTensor<kDev>(const_cast<void*>(slot_mapping.data()),
                         slot_mapping_shape_, slot_mapping_strides_,
                         slot_mapping_type_, device_index_);

  // `k_cache` / `v_cache` layout: `[num_blocks, num_kv_heads, block_size,
  // head_size]`. `block_size` is the third-to-last dimension.
  const auto block_size = static_cast<int64_t>(at_k_cache.size(-2));

  // Drop padding tokens (slot < 0) before scattering.
  auto valid_mask = at_slot_mapping.ge(0);
  auto valid_indices = at::nonzero(valid_mask).squeeze(-1);
  auto valid_slots = at_slot_mapping.index_select(0, valid_indices);
  auto block_idx = at::floor_divide(valid_slots, block_size);
  auto block_offset = valid_slots.remainder(block_size);

  auto k_valid = at_k.index_select(0, valid_indices);
  auto v_valid = at_v.index_select(0, valid_indices);

  using at::indexing::Slice;
  at_k_cache.index_put_({block_idx, Slice(), block_offset, Slice()}, k_valid);
  at_v_cache.index_put_({block_idx, Slice(), block_offset, Slice()}, v_valid);
}

template class Operator<PagedCaching, Device::Type::kCpu, 1>;
template class Operator<PagedCaching, Device::Type::kNvidia, 1>;
template class Operator<PagedCaching, Device::Type::kCambricon, 1>;
template class Operator<PagedCaching, Device::Type::kAscend, 1>;
template class Operator<PagedCaching, Device::Type::kMetax, 1>;
template class Operator<PagedCaching, Device::Type::kMoore, 1>;
template class Operator<PagedCaching, Device::Type::kIluvatar, 1>;
template class Operator<PagedCaching, Device::Type::kKunlun, 1>;
template class Operator<PagedCaching, Device::Type::kHygon, 1>;
template class Operator<PagedCaching, Device::Type::kQy, 1>;

}  // namespace infini::ops
