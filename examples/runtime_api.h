#ifndef INFINI_OPS_EXAMPLES_RUNTIME_API_H_
#define INFINI_OPS_EXAMPLES_RUNTIME_API_H_

#ifdef WITH_NVIDIA
#include <cuda_runtime.h>
#define DEVICE_MALLOC cudaMalloc
#define DEVICE_FREE cudaFree
#define DEVICE_MEMCPY cudaMemcpy
#define DEVICE_MEMSET cudaMemset
#define DEVICE_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define DEFAULT_DEVICE_TYPE Device::Type::kNvidia
#elif WITH_METAX
#include <mcr/mc_runtime.h>
#define DEVICE_MALLOC mcMalloc
#define DEVICE_FREE mcFree
#define DEVICE_MEMCPY mcMemcpy
#define DEVICE_MEMSET mcMemset
#define DEVICE_MEMCPY_HOST_TO_DEVICE mcMemcpyHostToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST mcMemcpyDeviceToHost
#define DEFAULT_DEVICE_TYPE Device::Type::kMetax
#elif WITH_CPU
#include <cstdlib>
#include <cstring>
#define DEVICE_MALLOC(ptr, size) (*(ptr) = std::malloc(size))
#define DEVICE_FREE std::free
#define DEVICE_MEMCPY(dst, src, size, kind) std::memcpy(dst, src, size)
#define DEVICE_MEMSET std::memset
#define DEVICE_MEMCPY_HOST_TO_DEVICE 0
#define DEVICE_MEMCPY_DEVICE_TO_HOST 1
#define DEFAULT_DEVICE_TYPE Device::Type::kCpu
#endif

#endif
