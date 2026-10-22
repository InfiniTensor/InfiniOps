#ifndef INFINI_OPS_HYGON_PAD_KERNEL_H_
#define INFINI_OPS_HYGON_PAD_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/pad/kernel.h"

namespace infini::ops {
namespace cuda_pad_detail {
struct HygonBackend : Runtime<Device::Type::kHygon> {
  static constexpr auto PeekAtLastError = cudaPeekAtLastError;
  static constexpr auto GetErrorString = cudaGetErrorString;
};
}  // namespace cuda_pad_detail

template <>
class Operator<Pad, Device::Type::kHygon, 0>
    : public CudaPad<cuda_pad_detail::HygonBackend> {
 public:
  using CudaPad<cuda_pad_detail::HygonBackend>::CudaPad;
};
}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_PAD_KERNEL_H_
