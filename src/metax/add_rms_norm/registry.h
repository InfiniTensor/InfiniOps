#ifndef INFINI_OPS_METAX_ADD_RMS_NORM_REGISTRY_H_
#define INFINI_OPS_METAX_ADD_RMS_NORM_REGISTRY_H_

#include "base/add_rms_norm.h"

namespace infini::ops {

template <>
struct ActiveImplementationsImpl<AddRmsNorm, Device::Type::kMetax> {
  using type = List<0, 1>;
};

}  // namespace infini::ops

#endif
