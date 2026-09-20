#include <infini/ops/graph_capture.h>

#ifdef INFINI_OPS_WITH_ASCEND_TORCH_CAPTURE
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <cstdio>
#include <exception>
#include <mutex>

#include "torch/ascend/c10.h"
#endif

namespace infini::ops {

#ifdef INFINI_OPS_WITH_ASCEND_TORCH_CAPTURE
namespace {

class AscendGraphCaptureMemory final : public GraphCaptureMemory {
 public:
  AscendGraphCaptureMemory(c10::DeviceIndex device, aclrtStream stream)
      : device_{device}, pool_{nullptr, false} {
    // Use torch_npu's ID generator so our pools cannot collide with NPUGraph
    // or user-created MemPools in the same process. Only this runtime's stream
    // is redirected; other runtimes and ordinary PyTorch streams stay isolated.
    c10_npu::NPUCachingAllocator::beginAllocateToPool(
        device_, pool_.id(),
        [stream](aclrtStream candidate) { return candidate == stream; });
  }

  void EndCapture() override {
    if (active_) {
      c10_npu::NPUCachingAllocator::endAllocateToPool(device_, pool_.id());
      active_ = false;
    }
  }

  ~AscendGraphCaptureMemory() override {
    try {
      EndCapture();
      // The caller must destroy its graph executable before dropping this
      // owner. Until then empty_cache/OOM reclamation must not free the pool.
      c10_npu::NPUCachingAllocator::releasePool(device_, pool_.id());
    } catch (const std::exception& error) {
      // If ending capture failed, retain the pool instead of making memory
      // available while the allocator might still be redirecting into it.
      std::fprintf(stderr, "InfiniOps capture pool cleanup failed: %s\n",
                   error.what());
    } catch (...) {
      std::fprintf(stderr, "InfiniOps capture pool cleanup failed\n");
    }
  }

 private:
  c10::DeviceIndex device_;
  c10_npu::MemPool pool_;
  bool active_{true};
};

}  // namespace
#endif

std::shared_ptr<GraphCaptureMemory> BeginGraphCaptureMemory(
    const infini::rt::Device& device, void* stream) {
#ifdef INFINI_OPS_WITH_ASCEND_TORCH_CAPTURE
  if (device.type() == infini::rt::Device::Type::kAscend) {
    TORCH_CHECK(stream != nullptr,
                "Ascend graph capture requires an explicit runtime stream");
    const C10<Device::Type::kAscend>::StreamGuard guard{
        C10<Device::Type::kAscend>::GetStreamFromExternal(stream,
                                                          device.index())};
    // Pure-copy C++ graphs may reach here before any ATen allocation. Stream
    // initialization does not initialize the caching allocator's device pools.
    // Serialize this initialization across independent runtime instances.
    static std::mutex allocator_init_mutex;
    {
      const std::lock_guard<std::mutex> lock{allocator_init_mutex};
      if (!c10_npu::NPUCachingAllocator::get()->initialized()) {
        // The inline NPUCachingAllocator::init error macro references private
        // symbols missing from wheels. Use ACL and the public virtual API.
        uint32_t device_count = 0;
        const auto status = aclrtGetDeviceCount(&device_count);
        TORCH_CHECK(status == ACL_ERROR_NONE,
                    "aclrtGetDeviceCount for graph capture failed: ", status);
        c10_npu::NPUCachingAllocator::get()->init(
            static_cast<int>(device_count));
      }
    }
    return std::make_shared<AscendGraphCaptureMemory>(
        static_cast<c10::DeviceIndex>(device.index()), stream);
  }
#else
  (void)device;
  (void)stream;
#endif
  return nullptr;
}

}  // namespace infini::ops
