#ifndef INFINI_OPS_METAX_CAST_KERNEL_H_
#define INFINI_OPS_METAX_CAST_KERNEL_H_

#ifdef WITH_TORCH

#include "base/cast.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kMetax> : public Cast {
 public:
  using Cast::Cast;

  void operator()(const Tensor input, Tensor out) const override;
};

}  // namespace infini::ops

#endif

#endif
