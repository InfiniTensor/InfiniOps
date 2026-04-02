#ifndef INFINI_OPS_RUNTIME_H_
#define INFINI_OPS_RUNTIME_H_

#include <type_traits>

#include "device.h"

namespace infini::ops {

template <Device::Type device_type>
struct Runtime;

// ---- Interface enforcement via CRTP ----
//
// Inherit from the appropriate base to declare which interface level a
// Runtime specialization implements.  After the struct is fully defined,
// call `static_assert(Runtime<...>::Validate())` — the chained Validate()
// checks every required member's existence and signature at compile time,
// analogous to how `override` catches signature mismatches for virtual
// functions.
//
//   RuntimeBase        – kDeviceType only           (e.g. CPU)
//   DeviceRuntime      – + Stream, Malloc, Free     (e.g. Cambricon)
//   CudaLikeRuntime    – + Memcpy, MemcpyHostToDevice, GetOptimalBlockSize

/// Every Runtime must provide `static constexpr Device::Type kDeviceType`.
template <typename Derived>
struct RuntimeBase {
  static constexpr bool Validate() {
    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Derived::kDeviceType)>,
                       Device::Type>,
        "Runtime must define 'static constexpr Device::Type kDeviceType'");
    return true;
  }
};

/// Runtimes with device memory must additionally provide Stream, Malloc, Free.
template <typename Derived>
struct DeviceRuntime : RuntimeBase<Derived> {
  static constexpr bool Validate() {
    RuntimeBase<Derived>::Validate();
    static_assert(sizeof(typename Derived::Stream) > 0,
                  "Runtime must define a 'Stream' type alias");
    static_assert(
        std::is_invocable_v<decltype(Derived::Malloc), void**, size_t>,
        "Runtime::Malloc must be callable with (void**, size_t)");
    static_assert(std::is_invocable_v<decltype(Derived::Free), void*>,
                  "Runtime::Free must be callable with (void*)");
    return true;
  }
};

/// CUDA-like runtimes must additionally provide Memcpy, MemcpyHostToDevice,
/// and GetOptimalBlockSize.
template <typename Derived>
struct CudaLikeRuntime : DeviceRuntime<Derived> {
  static constexpr bool Validate() {
    DeviceRuntime<Derived>::Validate();
    static_assert(
        std::is_invocable_v<decltype(Derived::Memcpy), void*, const void*,
                            size_t, decltype(Derived::MemcpyHostToDevice)>,
        "Runtime::Memcpy must be callable with "
        "(void*, const void*, size_t, MemcpyHostToDevice)");
    static_assert(std::is_same_v<decltype(Derived::GetOptimalBlockSize()), int>,
                  "Runtime::GetOptimalBlockSize() must return int");
    return true;
  }
};

}  // namespace infini::ops

#endif
