#pragma once
#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"
namespace infini::ops {
struct MambaScanMetaxBackend : Runtime<Device::Type::kMetax> {
  static void CheckLaunch() {}
};
template <>
class Operator<MambaSelectiveScan, Device::Type::kMetax>
    : public CudaMambaSelectiveScan<MambaScanMetaxBackend> {
 public:
  using CudaMambaSelectiveScan<MambaScanMetaxBackend>::CudaMambaSelectiveScan;
};
}  // namespace infini::ops
