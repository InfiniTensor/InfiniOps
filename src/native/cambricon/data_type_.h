#ifndef INFINI_OPS_CAMBRICON_DATA_TYPE__H_
#define INFINI_OPS_CAMBRICON_DATA_TYPE__H_

#include "bang_bf16.h"
#include "bang_fp16.h"
#include "cambricon/device_.h"
#include "data_type.h"

namespace infini::ops {

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kBFloat16> {
  using type = __bang_bfloat16;
};

}  // namespace infini::ops

#endif
