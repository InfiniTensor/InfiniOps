#pragma once
#include "base/cast.h"
namespace infini::ops {
template <Device::Type kDev>
class Operator<Cast, kDev, 8> : public Cast {
 public:
  using Cast::Cast;
  void operator()(const Tensor input, Tensor out) const override;
};
}
