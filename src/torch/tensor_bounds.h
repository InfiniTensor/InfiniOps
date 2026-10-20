#ifndef INFINI_OPS_TORCH_TENSOR_BOUNDS_H_
#define INFINI_OPS_TORCH_TENSOR_BOUNDS_H_
#include <cstdint>
#include <limits>

#include "tensor.h"
namespace infini::ops::torch_detail {
inline bool AddressRange(const Tensor t, std::uintptr_t& first,
                         std::uintptr_t& last) {
  first = reinterpret_cast<std::uintptr_t>(t.data());
  if (t.numel() == 0) {
    last = first;
    return true;
  }
  std::size_t span = 1;
  const auto limit = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < t.ndim(); ++i) {
    if (t.stride(i) < 0) return false;
    const auto size = t.size(i) - 1;
    const auto stride = static_cast<std::size_t>(t.stride(i));
    if (size && stride > (limit - span) / size) return false;
    span += size * stride;
  }
  if (!first || span > (std::numeric_limits<std::uintptr_t>::max() - first) /
                           t.element_size())
    return false;
  last = first + span * t.element_size();
  return true;
}
inline bool Disjoint(const Tensor a, const Tensor b) {
  std::uintptr_t ab, ae, bb, be;
  return AddressRange(a, ab, ae) && AddressRange(b, bb, be) &&
         (ab == ae || bb == be || ae <= bb || be <= ab);
}
inline bool SameView(const Tensor a, const Tensor b) {
  return a.data() == b.data() && a.dtype() == b.dtype() &&
         a.shape() == b.shape() && a.strides() == b.strides();
}
}  // namespace infini::ops::torch_detail
#endif
