#ifndef INFINI_OPS_ASCEND_RUNTIME__H_
#define INFINI_OPS_ASCEND_RUNTIME__H_

// clang-format off
#include "acl/acl.h"
// clang-format on

#include "ascend/device_.h"
#include "runtime.h"

namespace infini::ops {

template <>
struct Runtime<Device::Type::kAscend>
    : DeviceRuntime<Runtime<Device::Type::kAscend>> {
  using Stream = aclrtStream;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  static constexpr auto Malloc = [](void** ptr, size_t size) {
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  };

  static constexpr auto Free = aclrtFree;

  static constexpr auto Memcpy = aclrtMemcpy;

  static constexpr auto MemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto MemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto Memset = aclrtMemset;
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::ops

#endif
