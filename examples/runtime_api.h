#ifndef INFINI_OPS_EXAMPLES_RUNTIME_API_H_
#define INFINI_OPS_EXAMPLES_RUNTIME_API_H_

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "device.h"

#ifdef WITH_NVIDIA
#include "nvidia/gemm/cublas.h"
#include "nvidia/runtime_.h"
#elif WITH_ILUVATAR
#include "iluvatar/gemm/cublas.h"
#include "iluvatar/runtime_.h"
#elif WITH_METAX
#include "metax/gemm/mcblas.h"
#include "metax/runtime_.h"
#elif WITH_CAMBRICON
#include "cambricon/gemm/cnblas.h"
#elif WITH_MOORE
#include "moore/gemm/mublas.h"
#include "moore/runtime_.h"
#elif WITH_CPU
#include "cpu/gemm/gemm.h"
#else
#error "One `WITH_*` backend must be enabled for the examples."
#endif

namespace infini::ops {

template <Device::Type device_type>
struct ExampleRuntimeUtils;

#ifdef WITH_NVIDIA

template <>
struct ExampleRuntimeUtils<Device::Type::kNvidia> {
  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static auto Malloc(void** ptr, std::size_t size) {
    return Runtime<kDeviceType>::Malloc(ptr, size);
  }

  static auto Free(void* ptr) { return Runtime<kDeviceType>::Free(ptr); }

  static auto Memcpy(void* dst, const void* src, std::size_t size,
                     cudaMemcpyKind kind) {
    return Runtime<kDeviceType>::Memcpy(dst, src, size, kind);
  }

  static auto Memset(void* ptr, int value, std::size_t size) {
    return cudaMemset(ptr, value, size);
  }

  static constexpr auto kMemcpyHostToDevice =
      Runtime<kDeviceType>::MemcpyHostToDevice;
  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kNvidia>;

#elif WITH_ILUVATAR

template <>
struct ExampleRuntimeUtils<Device::Type::kIluvatar> {
  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static auto Malloc(void** ptr, std::size_t size) {
    return Runtime<kDeviceType>::Malloc(ptr, size);
  }

  static auto Free(void* ptr) { return Runtime<kDeviceType>::Free(ptr); }

  static auto Memcpy(void* dst, const void* src, std::size_t size,
                     cudaMemcpyKind kind) {
    return Runtime<kDeviceType>::Memcpy(dst, src, size, kind);
  }

  static auto Memset(void* ptr, int value, std::size_t size) {
    return cudaMemset(ptr, value, size);
  }

  static constexpr auto kMemcpyHostToDevice =
      Runtime<kDeviceType>::MemcpyHostToDevice;
  static constexpr auto kMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kIluvatar>;

#elif WITH_METAX

template <>
struct ExampleRuntimeUtils<Device::Type::kMetax> {
  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static auto Malloc(void** ptr, std::size_t size) {
    return Runtime<kDeviceType>::Malloc(ptr, size);
  }

  static auto Free(void* ptr) { return Runtime<kDeviceType>::Free(ptr); }

  static auto Memcpy(void* dst, const void* src, std::size_t size,
                     mcMemcpyKind_t kind) {
    return Runtime<kDeviceType>::Memcpy(dst, src, size, kind);
  }

  static auto Memset(void* ptr, int value, std::size_t size) {
    return mcMemset(ptr, value, size);
  }

  static constexpr auto kMemcpyHostToDevice =
      Runtime<kDeviceType>::MemcpyHostToDevice;
  static constexpr auto kMemcpyDeviceToHost = mcMemcpyDeviceToHost;
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kMetax>;

#elif WITH_CAMBRICON

template <>
struct ExampleRuntimeUtils<Device::Type::kCambricon> {
  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

  static auto Malloc(void** ptr, std::size_t size) {
    return cnrtMalloc(ptr, size);
  }

  static auto Free(void* ptr) { return cnrtFree(ptr); }

  static auto Memcpy(void* dst, const void* src, std::size_t size,
                     cnrtMemcpyType_t kind) {
    return cnrtMemcpy(dst, src, size, kind);
  }

  static auto Memset(void* ptr, int value, std::size_t size) {
    return cnrtMemset(ptr, value, size);
  }

  static constexpr auto kMemcpyHostToDevice = cnrtMemcpyHostToDev;
  static constexpr auto kMemcpyDeviceToHost = cnrtMemcpyDevToHost;
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kCambricon>;

#elif WITH_MOORE

template <>
struct ExampleRuntimeUtils<Device::Type::kMoore> {
  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static auto Malloc(void** ptr, std::size_t size) {
    return Runtime<kDeviceType>::Malloc(ptr, size);
  }

  static auto Free(void* ptr) { return Runtime<kDeviceType>::Free(ptr); }

  static auto Memcpy(void* dst, const void* src, std::size_t size,
                     musaMemcpyKind kind) {
    return Runtime<kDeviceType>::Memcpy(dst, src, size, kind);
  }

  static auto Memset(void* ptr, int value, std::size_t size) {
    return musaMemset(ptr, value, size);
  }

  static constexpr auto kMemcpyHostToDevice =
      Runtime<kDeviceType>::MemcpyHostToDevice;
  static constexpr auto kMemcpyDeviceToHost = musaMemcpyDeviceToHost;
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kMoore>;

#elif WITH_CPU

template <>
struct ExampleRuntimeUtils<Device::Type::kCpu> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;
  static constexpr int kMemcpyHostToDevice = 0;
  static constexpr int kMemcpyDeviceToHost = 1;

  static void Malloc(void** ptr, std::size_t size) { *ptr = std::malloc(size); }

  static void Free(void* ptr) { std::free(ptr); }

  static void Memcpy(void* dst, const void* src, std::size_t size, int) {
    std::memcpy(dst, src, size);
  }

  static void Memset(void* ptr, int value, std::size_t size) {
    std::memset(ptr, value, size);
  }
};

using DefaultRuntimeUtils = ExampleRuntimeUtils<Device::Type::kCpu>;

#endif

}  // namespace infini::ops

#endif
