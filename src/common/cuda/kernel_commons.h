#ifndef INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_
#define INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_

#ifdef WITH_NVIDIA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#elif defined(WITH_ILUVATAR)
#include <cuda_runtime.h>
#elif WITH_METAX  // TODO: Use `defined`.
#include <mcr/mc_runtime.h>
#endif

#include "cast.h"

namespace infini::ops {

// Block size constants for different GPU architectures
constexpr int CUDA_BLOCK_SIZE_128 = 128;
constexpr int CUDA_BLOCK_SIZE_256 = 256;
constexpr int CUDA_BLOCK_SIZE_512 = 512;
constexpr int CUDA_BLOCK_SIZE_1024 = 1024;

// Query the maximum threads per block for the current CUDA device
inline int queryMaxThreadsPerBlock() {
#ifdef WITH_NVIDIA
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerBlock;
#elif WITH_METAX
  // TODO: Add MCR device properties query for Metax
  return CUDA_BLOCK_SIZE_256;  // Default fallback
#endif
}

// Get optimal block size based on GPU hardware architecture
inline int getOptimalBlockSize() {
  int max_threads = queryMaxThreadsPerBlock();

  // Select the largest supported block size for better performance
  // Prioritize larger blocks for better GPU utilization
  if (max_threads >= CUDA_BLOCK_SIZE_1024) {
    return CUDA_BLOCK_SIZE_1024;
  } else if (max_threads >= CUDA_BLOCK_SIZE_512) {
    return CUDA_BLOCK_SIZE_512;
  } else if (max_threads >= CUDA_BLOCK_SIZE_256) {
    return CUDA_BLOCK_SIZE_256;
  } else {
    return CUDA_BLOCK_SIZE_128;  // Fallback to minimum
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
