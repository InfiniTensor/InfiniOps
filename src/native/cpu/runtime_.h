#ifndef INFINI_OPS_CPU_RUNTIME_H_
#define INFINI_OPS_CPU_RUNTIME_H_

#include <cstdlib>
#include <cstring>

#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  static void Malloc(void** ptr, std::size_t size) { *ptr = std::malloc(size); }

  static void Free(void* ptr) { std::free(ptr); }

  static void Memcpy(void* dst, const void* src, std::size_t size, int) {
    std::memcpy(dst, src, size);
  }

  static constexpr auto Memset = std::memset;

  static constexpr int MemcpyHostToDevice = 0;

  static constexpr int MemcpyDeviceToHost = 1;
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::ops

#endif
