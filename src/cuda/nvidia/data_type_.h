#ifndef INFINI_OPS_NVIDIA_DATA_TYPE__H_
#define INFINI_OPS_NVIDIA_DATA_TYPE__H_

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// clang-format on

#include "cuda/nvidia/device_.h"
#include "data_type.h"

namespace infini::ops {

using cuda_bfloat16 = nv_bfloat16;

using cuda_bfloat162 = nv_bfloat162;

template <>
struct TypeMap<Device::Type::kNvidia, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kNvidia, DataType::kBFloat16> {
  using type = __nv_bfloat16;
};

}  // namespace infini::ops

#endif
