#ifndef INFINI_OPS_METAX_PAD_KERNEL_H_
#define INFINI_OPS_METAX_PAD_KERNEL_H_

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/pad/kernel.h"

namespace infini::ops {
namespace cuda_pad_detail {
struct MetaxBackend : Runtime<Device::Type::kMetax> {
  static constexpr auto PeekAtLastError = mcPeekAtLastError;
  static constexpr auto GetErrorString = mcGetErrorString;
};
}  // namespace cuda_pad_detail

template <>
class Operator<Pad, Device::Type::kMetax, 0>
    : public CudaPad<cuda_pad_detail::MetaxBackend> {
 public:
  using CudaPad<cuda_pad_detail::MetaxBackend>::CudaPad;
};
}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_PAD_KERNEL_H_
