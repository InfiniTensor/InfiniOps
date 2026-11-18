#ifndef INFINI_OPS_TORCH_STORAGE_OVERLAP_H_
#define INFINI_OPS_TORCH_STORAGE_OVERLAP_H_

#include <ATen/ATen.h>

#include <cstdint>

namespace infini::ops {

// External pointers have independent ATen Storage identities. Include stride
// gaps in the byte span, conservatively treating interleaved views as overlap.
inline bool StorageSpansOverlap(const at::Tensor& a, const at::Tensor& b) {
  if (a.numel() == 0 || b.numel() == 0) return false;
  const auto ap = reinterpret_cast<std::uintptr_t>(a.data_ptr());
  const auto bp = reinterpret_cast<std::uintptr_t>(b.data_ptr());
  const auto an = a.storage().nbytes();
  const auto bn = b.storage().nbytes();
  return an && bn && (ap <= bp ? bp - ap < an : ap - bp < bn);
}

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_STORAGE_OVERLAP_H_
