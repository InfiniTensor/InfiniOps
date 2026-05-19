#ifndef INFINI_OPS_HYGON_DEVICE__H_
#define INFINI_OPS_HYGON_DEVICE__H_

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// clang-format on

#include "data_type.h"
#include "device.h"
#include "native/cuda/caster.cuh"

namespace infini::ops {

template <>
struct DeviceEnabled<Device::Type::kHygon> : std::true_type {};

// Some DTK toolchains expose the underlying bf16 structs but gate the
// nv_bfloat16 typedefs behind CUDA_NO_BFLOAT16.
using cuda_bfloat16 = __nv_bfloat16;

using cuda_bfloat162 = __nv_bfloat162;

namespace detail {

template <>
struct ToFloat<Device::Type::kHygon, half> {
  __host__ __device__ float operator()(half x) { return __half2float(x); }
};

template <>
struct ToFloat<Device::Type::kHygon, __nv_bfloat16> {
  __host__ __device__ float operator()(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct FromFloat<Device::Type::kHygon, half> {
  __host__ __device__ half operator()(float f) { return __float2half(f); }
};

template <>
struct FromFloat<Device::Type::kHygon, __nv_bfloat16> {
  __host__ __device__ __nv_bfloat16 operator()(float f) {
    return __float2bfloat16(f);
  }
};

}  // namespace detail

template <>
struct TypeMap<Device::Type::kHygon, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kHygon, DataType::kBFloat16> {
  using type = __nv_bfloat16;
};

template <>
struct Caster<Device::Type::kHygon> : CudaCasterImpl<Device::Type::kHygon> {};

}  // namespace infini::ops

#endif
