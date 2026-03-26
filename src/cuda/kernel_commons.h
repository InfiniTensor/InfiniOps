#ifndef INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_
#define INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_

#ifdef WITH_NVIDIA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#elif defined(WITH_ILUVATAR)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#elif defined(WITH_METAX)
#include <mcr/mc_runtime.h>
using cuda_bfloat16 = maca_bfloat16;
using cuda_bfloat162 = maca_bfloat162;
#elif defined(WITH_MOORE)
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
using cuda_bfloat16 = __mt_bfloat16;
using cuda_bfloat162 = __mt_bfloat162;
#endif

#include <cstdlib>
#include <iostream>
#include <vector>

#include "caster.h"

namespace infini::ops {

constexpr int CUDA_BLOCK_SIZE_128 = 128;
constexpr int CUDA_BLOCK_SIZE_256 = 256;
constexpr int CUDA_BLOCK_SIZE_512 = 512;
constexpr int CUDA_BLOCK_SIZE_1024 = 1024;
constexpr int CUDA_BLOCK_SIZE_2048 = 2048;

#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
// Cache `cudaDeviceProp` per device, initialized once at first access.
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

    if (device_id < 0 || device_id >= static_cast<int>(cache.size())) {
      std::cerr << "error: `device_id` " << device_id << " is out of range [0, "
                << cache.size() << ") in `GetDeviceProps`\n";
      std::abort();
    }
    return cache[device_id];
  }
};

inline int QueryMaxThreadsPerBlock() {
  return DevicePropertyCache::GetCurrentDeviceProps().maxThreadsPerBlock;
}
#elif defined(WITH_METAX)
inline int QueryMaxThreadsPerBlock() {
  // TODO: Add MCR device properties query for Metax.
  return CUDA_BLOCK_SIZE_256;
}
#elif defined(WITH_MOORE)
inline int QueryMaxThreadsPerBlock() {
  int device = 0;
  musaGetDevice(&device);
  musaDeviceProp prop;
  musaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerBlock;
}
#endif

// Get optimal block size based on GPU hardware architecture.
inline int GetOptimalBlockSize() {
  int max_threads = QueryMaxThreadsPerBlock();
  if (max_threads >= CUDA_BLOCK_SIZE_2048) {
    return CUDA_BLOCK_SIZE_2048;
  } else if (max_threads >= CUDA_BLOCK_SIZE_1024) {
    return CUDA_BLOCK_SIZE_1024;
  } else if (max_threads >= CUDA_BLOCK_SIZE_512) {
    return CUDA_BLOCK_SIZE_512;
  } else if (max_threads >= CUDA_BLOCK_SIZE_256) {
    return CUDA_BLOCK_SIZE_256;
  } else {
    return CUDA_BLOCK_SIZE_128;
  }
}

__forceinline__ __device__ __host__ size_t
IndexToOffset(size_t flat_index, size_t ndim, const size_t* shape,
              const ptrdiff_t* strides) {
  size_t res = 0;
  for (size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

}  // namespace infini::ops

#endif
