#ifndef INFINI_OPS_CPU_DATA_TYPE__H_
#define INFINI_OPS_CPU_DATA_TYPE__H_

#include "data_type.h"
#include "native/cpu/device_.h"

namespace infini::ops {

template <>
struct TypeMap<Device::Type::kCpu, DataType::kFloat16> {
  using type = Float16;
};

template <>
struct TypeMap<Device::Type::kCpu, DataType::kBFloat16> {
  using type = BFloat16;
};

}  // namespace infini::ops

#endif
