#ifndef INFINI_OPS_TORCH_STREAM__H_
#define INFINI_OPS_TORCH_STREAM__H_

#include <c10/core/StreamGuard.h>

#include <optional>

#if defined(WITH_NVIDIA) || defined(WITH_METAX)
#include <c10/cuda/CUDAStream.h>
#endif

#ifdef WITH_ASCEND
#include <torch_npu/csrc/core/npu/NPUStream.h>

#if __has_include(<torch_npu/csrc/core/npu/NPUStreamUtils.h>)
#define INFINI_OPS_HAS_TORCH_NPU_EXTERNAL_STREAM 1
#endif
#endif

#include "device.h"
#include "runtime.h"

namespace infini::ops::detail {

template <Device::Type kDev>
class TorchStreamGuard {
 public:
  TorchStreamGuard(void*, int) {}
};

#if defined(WITH_NVIDIA) || defined(WITH_METAX)
template <Device::Type kDev>
class CudaTorchStreamGuard {
 public:
  CudaTorchStreamGuard(void* stream, int device_index) {
    if (stream == nullptr) return;

    stream_guard_.emplace(c10::cuda::getStreamFromExternal(
        reinterpret_cast<typename Runtime<kDev>::Stream>(stream),
        static_cast<c10::DeviceIndex>(device_index)));
  }

 private:
  std::optional<c10::StreamGuard> stream_guard_;
};
#endif

#ifdef WITH_NVIDIA
template <>
class TorchStreamGuard<Device::Type::kNvidia>
    : public CudaTorchStreamGuard<Device::Type::kNvidia> {
 public:
  using CudaTorchStreamGuard<Device::Type::kNvidia>::CudaTorchStreamGuard;
};
#endif

#ifdef WITH_METAX
template <>
class TorchStreamGuard<Device::Type::kMetax>
    : public CudaTorchStreamGuard<Device::Type::kMetax> {
 public:
  using CudaTorchStreamGuard<Device::Type::kMetax>::CudaTorchStreamGuard;
};
#endif

#ifdef INFINI_OPS_HAS_TORCH_NPU_EXTERNAL_STREAM
template <>
class TorchStreamGuard<Device::Type::kAscend> {
 public:
  TorchStreamGuard(void* stream, int device_index) {
    if (stream == nullptr) return;

    stream_guard_.emplace(c10_npu::getStreamFromExternal(
        reinterpret_cast<Runtime<Device::Type::kAscend>::Stream>(stream),
        static_cast<c10::DeviceIndex>(device_index)));
  }

 private:
  std::optional<c10::StreamGuard> stream_guard_;
};
#endif

}  // namespace infini::ops::detail

#ifdef INFINI_OPS_HAS_TORCH_NPU_EXTERNAL_STREAM
#undef INFINI_OPS_HAS_TORCH_NPU_EXTERNAL_STREAM
#endif

#endif
