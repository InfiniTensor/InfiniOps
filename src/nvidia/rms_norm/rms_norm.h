#ifndef INFINI_OPS_NVIDIA_RMS_NORM_H_
#define INFINI_OPS_NVIDIA_RMS_NORM_H_

#include "base/rms_norm.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia> : public RmsNorm {
 public:
  Operator(const Tensor y, const Tensor x, const Tensor w, float epsilon)
      : RmsNorm{y, x, w, epsilon} {}

  Operator(const Tensor y, const Tensor x, const Tensor w)
      : Operator{y, x, w, 1e-6f} {}

  void operator()(void* stream, Tensor y, const Tensor x, const Tensor w,
                  float /*epsilon*/ = 0) const override;
};

}  // namespace infini::ops

#endif
