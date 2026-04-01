#ifndef INFINI_OPS_MOORE_DEVICE__H_
#define INFINI_OPS_MOORE_DEVICE__H_

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "cuda/caster_.h"
#include "data_type.h"
#include "device.h"

namespace infini::ops {

using cuda_bfloat16 = __mt_bfloat16;
using cuda_bfloat162 = __mt_bfloat162;

template <>
struct TypeMap<Device::Type::kMoore, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kMoore, DataType::kBFloat16> {
  using type = __mt_bfloat16;
};

inline int QueryMaxThreadsPerBlock() {
  int device = 0;
  musaGetDevice(&device);
  musaDeviceProp prop;
  musaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerBlock;
}

template <>
struct Caster<Device::Type::kMoore> : CudaCasterImpl<Device::Type::kMoore> {};

}  // namespace infini::ops

#endif
