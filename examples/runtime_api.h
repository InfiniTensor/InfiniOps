#ifndef INFINI_OPS_EXAMPLES_RUNTIME_API_H_
#define INFINI_OPS_EXAMPLES_RUNTIME_API_H_

#ifdef USE_NVIDIA
#include <cuda_runtime.h>
#define DEVICE_MALLOC cudaMalloc
#define DEVICE_FREE cudaFree
#define DEVICE_MEMCPY cudaMemcpy
#define DEVICE_MEMSET cudaSet
#define DEVICE_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define DEFAULT_DEVICE_TYPE Device::Type::kNvidia
#elif USE_METAX
#include <mcr/mc_runtime.h>
#define DEVICE_MALLOC mcMalloc
#define DEVICE_FREE mcFree
#define DEVICE_MEMCPY mcMemcpy
#define DEVICE_MEMSET mcMemset
#define DEVICE_MEMCPY_HOST_TO_DEVICE mcMemcpyHostToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST mcMemcpyDeviceToHost
#define DEFAULT_DEVICE_TYPE Device::Type::kMetax
#elif USE_CPU
#define DEFAULT_DEVICE_TYPE Device::Type::kCpu
#endif

#endif
