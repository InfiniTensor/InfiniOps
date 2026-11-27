#ifndef INFINI_OPS_HYGON_SEPARATE_TABLE_ROPE_KERNEL_H_
#define INFINI_OPS_HYGON_SEPARATE_TABLE_ROPE_KERNEL_H_
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/separate_table_rope/kernel.h"
namespace infini::ops {
template <>
class Operator<SeparateTableRope, Device::Type::kHygon>
    : public CudaSeparateTableRope<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSeparateTableRope<
      Runtime<Device::Type::kHygon>>::CudaSeparateTableRope;
  void operator()(const Tensor x, const Tensor p, const Tensor s,
                  const Tensor c, bool neox, Tensor y) const override {
    CudaSeparateTableRope<Runtime<Device::Type::kHygon>>::operator()(x, p, s, c,
                                                                     neox, y);
    if (x.numel() == 0) return;
    // Same native HYGON launch-error API used by the installed Argmax source.
    // Peek reports immediate launch failure; it does not synchronize or claim
    // that device data access / position bounds are valid.
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
};
}  // namespace infini::ops
#endif
