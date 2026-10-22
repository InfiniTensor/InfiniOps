#ifndef INFINI_OPS_NVIDIA_PAD_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAD_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/pad/kernel.h"

namespace infini::ops {
namespace cuda_pad_detail {
struct NvidiaBackend : Runtime<Device::Type::kNvidia> {
  static constexpr auto PeekAtLastError = cudaPeekAtLastError;
  static constexpr auto GetErrorString = cudaGetErrorString;
};
}  // namespace cuda_pad_detail

template <>
class Operator<Pad, Device::Type::kNvidia, 0>
    : public CudaPad<cuda_pad_detail::NvidiaBackend> {
 public:
  using CudaPad<cuda_pad_detail::NvidiaBackend>::CudaPad;
};
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_PAD_KERNEL_H_
