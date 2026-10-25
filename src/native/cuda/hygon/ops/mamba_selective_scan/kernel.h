#pragma once
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"
namespace infini::ops {
struct MambaScanHygonBackend : Runtime<Device::Type::kHygon> {
  static void CheckLaunch() {
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("MambaSelectiveScan CUDA launch failed");
  }
};
template <>
class Operator<MambaSelectiveScan, Device::Type::kHygon>
    : public CudaMambaSelectiveScan<MambaScanHygonBackend> {
 public:
  using CudaMambaSelectiveScan<MambaScanHygonBackend>::CudaMambaSelectiveScan;
};
}  // namespace infini::ops
