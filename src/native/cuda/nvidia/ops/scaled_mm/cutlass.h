#ifndef INFINI_OPS_NVIDIA_SCALED_MM_CUTLASS_H_
#define INFINI_OPS_NVIDIA_SCALED_MM_CUTLASS_H_

#include <optional>

#include "base/scaled_mm.h"

namespace infini::ops {

class CutlassScaledMm : public ScaledMm {
 public:
  CutlassScaledMm(const Tensor a, const Tensor b, const Tensor scale_a,
                  const Tensor scale_b, std::optional<Tensor> bias, Tensor out);

  void operator()(const Tensor a, const Tensor b, const Tensor scale_a,
                  const Tensor scale_b, std::optional<Tensor> bias,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

template <>
class Operator<ScaledMm, Device::Type::kNvidia, 0> : public CutlassScaledMm {
 public:
  using CutlassScaledMm::CutlassScaledMm;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_SCALED_MM_CUTLASS_H_
