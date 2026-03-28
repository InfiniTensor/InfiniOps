#ifndef INFINI_OPS_METAX_DEVICE_H_
#define INFINI_OPS_METAX_DEVICE_H_

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mcr/mc_runtime.h>

#include "cuda/caster_.h"
#include "data_type.h"
#include "device.h"

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

// TODO: Add MCR device properties query for Metax.
inline int QueryMaxThreadsPerBlock() { return 256; }

template <>
struct Caster<Device::Type::kMetax> : CudaCasterImpl<Device::Type::kMetax> {};

}  // namespace infini::ops

#endif
