#ifndef INFINI_OPS_TORCH_CAST_H_
#define INFINI_OPS_TORCH_CAST_H_

#include "base/cast.h"

namespace infini::ops {

template <Device::Type kDev>
class AtenCast : public Cast {
 public:
  using Cast::Cast;

  static bool supports(const Tensor input, const Tensor out) {
    return input.device().type() == kDev && input.device() == out.device() &&
           input.shape() == out.shape();
  }

  void operator()(const Tensor input, Tensor out) const override;
};

template <>
class Operator<Cast, Device::Type::kCpu, 8>
    : public AtenCast<Device::Type::kCpu> {
 public:
  using AtenCast<Device::Type::kCpu>::AtenCast;
};

template <>
class Operator<Cast, Device::Type::kNvidia, 8>
    : public AtenCast<Device::Type::kNvidia> {
 public:
  using AtenCast<Device::Type::kNvidia>::AtenCast;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_CAST_H_
