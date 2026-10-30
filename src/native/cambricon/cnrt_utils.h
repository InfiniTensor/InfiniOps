#ifndef INFINI_OPS_CAMBRICON_CNRT_UTILS_H_
#define INFINI_OPS_CAMBRICON_CNRT_UTILS_H_

#include <cnrt.h>

#include <cstddef>
#include <memory>

namespace infini::ops::cnrt_utils {

struct DeviceBufferDeleter {
  using pointer = void*;

  void operator()(pointer buffer) const noexcept {
    if (buffer) {
      (void)cnrtFree(buffer);
    }
  }
};

using DeviceBuffer = std::unique_ptr<void, DeviceBufferDeleter>;

inline DeviceBuffer AllocateDeviceBuffer(std::size_t size) {
  if (size == 0) {
    return {};
  }

  void* buffer{nullptr};
  CNRT_CHECK(cnrtMalloc(&buffer, size));

  return DeviceBuffer{buffer};
}

// Called only when an invocation has no caller-owned workspace. Keep the
// fallback alive for asynchronous execution and reuse it on later calls.
inline void* DefaultWorkspace(DeviceBuffer& buffer, std::size_t size) {
  if (!buffer && size != 0) {
    buffer = AllocateDeviceBuffer(size);
  }
  return buffer.get();
}

}  // namespace infini::ops::cnrt_utils

#endif
