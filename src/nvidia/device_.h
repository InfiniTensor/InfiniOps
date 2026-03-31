#ifndef INFINI_OPS_NVIDIA_DEVICE_H_
#define INFINI_OPS_NVIDIA_DEVICE_H_

#include <cassert>
#include <vector>

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// clang-format on

#include "cuda/caster_.h"
#include "data_type.h"
#include "device.h"

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

// Caches `cudaDeviceProp` per device, initialized once at first access.
class DevicePropertyCache {
 public:
  static const cudaDeviceProp& GetCurrentDeviceProps() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    return GetDeviceProps(device_id);
  }

  static const cudaDeviceProp& GetDeviceProps(int device_id) {
    static std::vector<cudaDeviceProp> cache = []() {
      int count = 0;
      cudaGetDeviceCount(&count);
      if (count == 0) return std::vector<cudaDeviceProp>{};
      std::vector<cudaDeviceProp> props(count);
      for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&props[i], i);
      }
      return props;
    }();

    assert(device_id >= 0 && device_id < static_cast<int>(cache.size()));
    return cache[device_id];
  }
};

inline int QueryMaxThreadsPerBlock() {
  return DevicePropertyCache::GetCurrentDeviceProps().maxThreadsPerBlock;
}

template <>
struct Caster<Device::Type::kNvidia> : CudaCasterImpl<Device::Type::kNvidia> {};

}  // namespace infini::ops

#endif
