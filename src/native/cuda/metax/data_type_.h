#ifndef INFINI_OPS_METAX_DATA_TYPE__H_
#define INFINI_OPS_METAX_DATA_TYPE__H_

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mcr/mc_runtime.h>

#include "data_type.h"
#include "native/cuda/metax/device_.h"

namespace infini::ops {

using cuda_bfloat16 = maca_bfloat16;

using cuda_bfloat162 = maca_bfloat162;

template <>
struct TypeMap<Device::Type::kMetax, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kMetax, DataType::kBFloat16> {
  using type = __maca_bfloat16;
};

}  // namespace infini::ops

#endif
